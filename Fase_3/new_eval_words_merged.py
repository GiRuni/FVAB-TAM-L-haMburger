"""
Evaluate TAM heatmaps on a custom test set of images with:
  - binary object masks  (masks/<stem>/<obj>.png)
  - spatial relation config  (rel_config.yaml)

Metrics computed per token step:
  obj_iou   : Otsu-thresholded IoU against object binary mask  
  func_iou  : fraction of heatmap below noun-fg threshold      
  f1_iou    : harmonic mean of obj_iou and func_iou
  io_ratio  : fraction of heatmap mass inside the GT mask      
  wdp       : distance-weighted penalty outside GT mask        
  iou_hard  : hard IoU at 0.5 threshold                       

Models supported: Qwen2-VL, Qwen2.5-VL, InternVL3, LLaVA
Logit-lens mode: set --all_layers to run across all layers.
"""

import csv
import concurrent.futures
import json
import os
import re
import sys
import math
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt as _edt

warnings.filterwarnings("ignore")


def _norm_word(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _canonical_image_id(s: str) -> str:
    s = s.strip()
    if re.fullmatch(r"\d+", s):
        return str(int(s))
    return s


def _strip_leading_enumeration(line: str) -> str:
    """Remove leading numbering like '1. ' (possibly repeated)."""
    out = line.strip()
    while True:
        new_out = re.sub(r"^\s*\d+\.\s*", "", out)
        if new_out == out:
            break
        out = new_out
    return out.strip()


def load_object_word_queries(path: Path) -> dict:
    """
    Parse a text file with blocks:
      image_id
      object + attribute
      object + action
      object + relation
    Missing lines are allowed.
    """
    if not path.exists():
        return {}

    # utf-8-sig strips BOM if present (common in manually edited txt files)
    raw_lines = path.read_text(encoding="utf-8-sig").splitlines()

    out = {}
    current_id = None
    current_entries = []
    kinds = ["attribute", "action", "relation"]

    def flush_current():
        nonlocal current_id, current_entries
        if current_id is not None and current_entries:
            out[_canonical_image_id(current_id)] = current_entries[:]

    for ln in raw_lines:
        line = ln.strip()

        # New image block starts with lines like "18. 000000301867".
        image_match = re.fullmatch(r"\d+\.\s+(\d+)", line)
        if image_match:
            flush_current()
            current_id = _canonical_image_id(image_match.group(1))
            current_entries = []
            continue

        if current_id is None:
            continue

        relation_match = re.fullmatch(r"\s*(\d+)\.\s*(.*)", ln)
        if not relation_match:
            continue

        relation_num = relation_match.group(1)
        relation_text = relation_match.group(2).strip()
        if not relation_text:
            continue

        left = ""
        right = ""
        if "+" in relation_text:
            left, right = [x.strip() for x in relation_text.split("+", 1)]
        else:
            # Support malformed rows like: "person standing"
            parts = relation_text.split()
            if len(parts) >= 2:
                left = parts[0].strip()
                right = " ".join(parts[1:]).strip()

        if not left or not right:
            continue

        # Remove ID numbers from parentheses: "red (442619)" -> "red"
        right = re.sub(r"\s*\([^)]*\)\s*", "", right).strip()
        # Normalize spaces around relation phrases (e.g. "+behind").
        right = re.sub(r"\s+", " ", right).strip()
        
        if not right:
            continue
        
        kind = kinds[min(len(current_entries), len(kinds) - 1)]
        current_entries.append({
            "kind": kind,
            "object": left,
            "word": right,
            "mask_id": relation_num,
        })

    flush_current()
    return out


def _find_target_final_step(step_word_map: dict, target_word: str, token_labels: list) -> tuple[int, list[str]]:
    target_n = _norm_word(target_word)
    if not target_n:
        return -1, []

    def _match_token_span(parts: list[str]) -> tuple[int, list[str]]:
        """Return the last token index and matched labels for a contiguous span."""
        if not parts:
            return -1, []

        for i in range(len(token_labels) - len(parts) + 1):
            match = True
            for j, part in enumerate(parts):
                if _norm_word(token_labels[i + j]) != part:
                    match = False
                    break
            if match:
                return i + len(parts) - 1, token_labels[i:i + len(parts)]
        return -1, []

    # Check if this is a spatial relation phrase (contains "+")
    if "+" in target_word:
        # Split by "+" and normalize each part
        parts = [_norm_word(p.strip()) for p in target_word.split("+")]
        parts = [p for p in parts if p]  # Remove empty parts
        
        if len(parts) > 1:
            # Try to find consecutive matching tokens in the written order first.
            i, matched_labels = _match_token_span(parts)
            if i >= 0:
                print(f"    [MATCH SPATIAL] target_word='{target_word}' -> steps {i-len(parts)+1}-{i} (tokens: {matched_labels})")
                return i, matched_labels

            # Some generations flip the order of the same words (for example: 'brown bear').
            reversed_parts = list(reversed(parts))
            if reversed_parts != parts:
                i, matched_labels = _match_token_span(reversed_parts)
                if i >= 0:
                    print(f"    [MATCH SPATIAL REVERSED] target_word='{target_word}' -> steps {i-len(parts)+1}-{i} (tokens: {matched_labels})")
                    return i, matched_labels
            
            # Fallback: try to find any part of the spatial relation (including morphological variants)
            # Try matching the most significant part (usually the last one, the object/descriptor)
            most_significant = parts[-1] if len(parts) > 1 else parts[0]
            best_match = -1
            
            for i in range(len(token_labels)):
                tok_norm = _norm_word(token_labels[i])
                # Exact match
                if tok_norm == most_significant:
                    best_match = i
                    print(f"    [MATCH SPATIAL EXACT] target_word='{target_word}' -> step {i} (token: {token_labels[i]})")
                    return i
                # Morphological match: check if they share a common stem
                # e.g., "skis" matches "skiing" (both contain "ski")
                elif (len(tok_norm) >= 3 and len(most_significant) >= 3 and 
                      (tok_norm.startswith(most_significant[:3]) or most_significant.startswith(tok_norm[:3]))):
                    if best_match == -1:  # Keep first partial match
                        best_match = i
            
            if best_match >= 0:
                print(f"    [MATCH SPATIAL MORPHO] target_word='{target_word}' -> step {best_match} (token: {token_labels[best_match]})")
                return best_match, [token_labels[best_match]]

    candidates = []
    for step, meta in step_word_map.items():
        w = _norm_word(meta.get("word", ""))
        if w == target_n:
            candidates.append(int(meta.get("word_step_end", step)))
    if candidates:
        result = max(candidates)
        print(f"    [MATCH] target_word='{target_word}' (norm='{target_n}') -> step {result} in step_word_map")
        return result, [step_word_map[result]["word"]] if result in step_word_map else []

    # Fallback: exact token label match.
    last = -1
    for i, t in enumerate(token_labels):
        if _norm_word(t) == target_n:
            last = i
    
    if last >= 0:
        print(f"    [FALLBACK] target_word='{target_word}' (norm='{target_n}') -> step {last} in token_labels")
        return last, [token_labels[last]]
    else:
        print(f"    [NOT FOUND] target_word='{target_word}' (norm='{target_n}')")
        print(f"              step_word_map keys: {list(step_word_map.keys())}")
        print(f"              step_word_map words: {[_norm_word(m.get('word', '')) for m in step_word_map.values()]}")
        print(f"              token_labels: {token_labels}")
    return last, []


# ---------------------------------------------------------------------------
# Spatial config loading
# ---------------------------------------------------------------------------

def load_spatial_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    prep_config   = cfg["prepositions"]
    single_lookup = cfg["single_token_lookup"]
    multi_phrases = sorted(
        cfg["multi_token_phrases"],
        key=lambda p: len(p["phrase"]),
        reverse=True,   # greedy: longest first
    )
    spatial_tokens = set(single_lookup.keys())
    for p in multi_phrases:
        spatial_tokens.update(p["phrase"].lower().split())
    return {
        "prepositions":   prep_config,
        "single_lookup":  single_lookup,
        "multi_phrases":  multi_phrases,
        "spatial_tokens": spatial_tokens,
    }


# ---------------------------------------------------------------------------
# Mask utilities
# ---------------------------------------------------------------------------

def load_binary_mask(mask_path: Path) -> np.ndarray:
    """Load a binary PNG mask, returns uint8 array with values 0/1."""
    arr = np.array(Image.open(mask_path).convert("L"))
    return (arr > 0).astype(np.uint8)


def resize_mask(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    im = Image.fromarray(mask * 255).resize((w, h), resample=Image.NEAREST)
    return (np.array(im) > 0).astype(np.uint8)


def get_object_masks(stem: str, masks_root: Path) -> dict:
    """Returns {mask_id: np.ndarray} for all masks found for this image."""
    obj_dir = masks_root / stem
    if not obj_dir.exists():
        return {}
    masks = {}
    for p in obj_dir.iterdir():
        if p.suffix.lower() == ".png":
            masks[p.stem] = load_binary_mask(p)
    return masks


# ---------------------------------------------------------------------------
# Spatial relation mask strategies  (from notebook cell 15)
# ---------------------------------------------------------------------------

def _binary_dilate(mask: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return mask.astype(np.uint8)
    h, w = mask.shape
    out = np.zeros((h, w), dtype=np.uint8)
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return out
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy * dy + dx * dx > r * r:
                continue
            out[np.clip(ys + dy, 0, h - 1), np.clip(xs + dx, 0, w - 1)] = 1
    return out


def _mask_bbox(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None
    return (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max()))


def _union_bbox_mask(masks, h, w):
    bboxes = [b for b in [_mask_bbox(m) for m in masks if m is not None] if b is not None]
    if not bboxes:
        return np.ones((h, w), dtype=np.uint8)
    y0 = min(b[0] for b in bboxes); x0 = min(b[1] for b in bboxes)
    y1 = max(b[2] for b in bboxes); x1 = max(b[3] for b in bboxes)
    m = np.zeros((h, w), dtype=np.uint8)
    m[y0:y1 + 1, x0:x1 + 1] = 1
    return m


def relation_region_mask(canonical: str, sub_mask: np.ndarray,
                          obj_mask: np.ndarray, prep_config: dict) -> np.ndarray:
    prep_data = prep_config.get(canonical, {
        "mask_strategy": "contact_zone",
        "mask_params": {"dilation_px": 18, "focus": "any"},
    })
    strategy = prep_data.get("mask_strategy", "contact_zone")
    params   = prep_data.get("mask_params", {})
    r        = int(params.get("dilation_px", 18))
    focus    = params.get("focus", "any")
    h, w     = sub_mask.shape

    sub_d = _binary_dilate(sub_mask, r)
    obj_d = _binary_dilate(obj_mask, r)
    b_obj = _mask_bbox(obj_mask)
    b_sub = _mask_bbox(sub_mask)

    def nonempty(m):
        return m if m.sum() > 0 else _union_bbox_mask([sub_mask, obj_mask], h, w)

    if strategy == "contact_zone":
        contact = (sub_d & obj_d).astype(np.uint8)
        if contact.sum() > 0:
            return contact
        if b_obj and b_sub:
            y0o, x0o, y1o, x1o = b_obj
            strip_h = max(1, int((y1o - y0o + 1) * 0.30))
            strip = np.zeros((h, w), dtype=np.uint8)
            if focus == "top_of_object":
                strip[y0o:y0o + strip_h, x0o:x1o + 1] = 1
            elif focus == "bottom_of_object":
                strip[max(y0o, y1o - strip_h + 1):y1o + 1, x0o:x1o + 1] = 1
            else:
                strip[y0o:y0o + strip_h, x0o:x1o + 1] = 1
                strip[max(y0o, y1o - strip_h + 1):y1o + 1, x0o:x1o + 1] = 1
            cand = (sub_d & strip).astype(np.uint8)
            if cand.sum() > 0:
                return cand
        return nonempty(np.clip(sub_d.astype(np.int32) + obj_d.astype(np.int32), 0, 1).astype(np.uint8))

    elif strategy == "object_mask":
        return nonempty(obj_mask.astype(np.uint8))
    elif strategy == "subject_mask":
        return nonempty(sub_mask.astype(np.uint8))
    elif strategy == "between_zone":
        if not b_obj or not b_sub:
            return nonempty((sub_d & obj_d).astype(np.uint8))
        ys0 = min(b_sub[0], b_obj[0]); xs0 = min(b_sub[1], b_obj[1])
        ys1 = max(b_sub[2], b_obj[2]); xs1 = max(b_sub[3], b_obj[3])
        region = np.zeros((h, w), dtype=np.uint8)
        region[ys0:ys1 + 1, xs0:xs1 + 1] = 1
        region[b_sub[0]:b_sub[2] + 1, b_sub[1]:b_sub[3] + 1] = 0
        region[b_obj[0]:b_obj[2] + 1, b_obj[1]:b_obj[3] + 1] = 0
        cand = (region & sub_d & obj_d).astype(np.uint8)
        return nonempty(cand if cand.sum() > 0 else region)
    elif strategy == "subject_outside_object":
        return nonempty((sub_d & (~obj_mask.astype(bool))).astype(np.uint8))
    else:
        return _union_bbox_mask([sub_mask, obj_mask], h, w)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _pnorm(x: np.ndarray, lo=1.0, hi=99.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    vlo, vhi = np.nanpercentile(x, lo), np.nanpercentile(x, hi)
    d = vhi - vlo
    if not np.isfinite(d) or d < 1e-12:
        return np.zeros_like(x)
    return np.clip((x - vlo) / d, 0.0, 1.0)


def metric_obj_iou_and_thresh(heatmap: np.ndarray, mask: np.ndarray):
    """Calcola sia l'IoU che la soglia Otsu usata dall'oggetto (serve per il func_iou)."""
    h, w = mask.shape
    hm = cv2.resize(heatmap, (w, h))
    t, pred = cv2.threshold(hm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    gt = mask.astype(np.uint8)
    if gt.sum() == 0:
        return float("nan"), t
    tp = float((gt * (pred > 0)).sum())
    obj_iou = tp / ((gt + pred / 255) > 0).sum()
    return obj_iou, t


def metric_func_iou(heatmap: np.ndarray, fg_thresh: float) -> float:
    """Implementazione classica del func_iou: frazione di heatmap inferiore alla soglia Otsu."""
    if heatmap.size == 0:
        return float("nan")
    return float((heatmap < fg_thresh).sum()) / heatmap.size


def metric_iou_hard(heatmap: np.ndarray, mask: np.ndarray,
                    thr: float = 0.5, eps: float = 1e-12) -> float:
    """Hard IoU at fixed threshold after percentile normalisation."""
    A = _pnorm(heatmap)
    h, w = mask.shape
    A = cv2.resize(A, (w, h))
    Ab = (A >= thr).astype(np.uint8)
    Mb = mask.astype(np.uint8)
    return float((Ab & Mb).sum()) / float((Ab | Mb).sum() + eps)


def metric_io_ratio(heatmap: np.ndarray, mask: np.ndarray, eps: float = 1e-12) -> float:
    """Fraction of heatmap mass inside GT mask."""
    A = _pnorm(heatmap).astype(np.float32)
    h, w = mask.shape
    A = cv2.resize(A, (w, h))
    M = mask.astype(np.float32)
    t = float(A.sum())
    return float((A * M).sum()) / (t + eps) if t > eps else 0.0


def metric_wdp(heatmap: np.ndarray, mask: np.ndarray, eps: float = 1e-12) -> float:
    """Distance-weighted penalty — activation outside GT mask."""
    A = _pnorm(heatmap).astype(np.float32)
    h, w = mask.shape
    A = cv2.resize(A, (w, h))
    M = (mask > 0).astype(np.uint8)
    t = float(A.sum())
    if t < eps:
        return 0.0
    d = _edt(1 - M).astype(np.float32) / max(math.sqrt(h * h + w * w), eps)
    return float((A * (1 - M.astype(np.float32)) * d).sum()) / (t + eps)


def compute_all_metrics(heatmap: np.ndarray, mask: np.ndarray) -> dict:
    obj_iou, fg_thresh = metric_obj_iou_and_thresh(heatmap, mask)
    hard = metric_iou_hard(heatmap, mask)
    io   = metric_io_ratio(heatmap, mask)
    wdp  = metric_wdp(heatmap, mask)
    
    # Calcoliamo le metriche classiche di eval.py
    func = metric_func_iou(heatmap, fg_thresh)
    
    # Calcoliamo il vero F1 classico
    f1 = float("nan")
    if not math.isnan(obj_iou) and not math.isnan(func) and (obj_iou + func) > 0:
        f1 = 2 * obj_iou * func / (obj_iou + func)
        
    return {
        "obj_iou": obj_iou, 
        "iou_hard": hard, 
        "io_ratio": io, 
        "wdp": wdp, 
        "func_iou": func, 
        "f1_iou": f1
    }


# ---------------------------------------------------------------------------
# Spatial token detection
# ---------------------------------------------------------------------------

def find_spatial_steps(token_labels: list, spatial_cfg: dict) -> list:
    """
    Returns [(step_idx, matched_phrase, canonical), ...].
    Supports multi-token phrases (greedy) before single tokens.
    """
    decoded = [t.strip().lower() for t in token_labels]
    out = []
    i = 0
    while i < len(decoded):
        matched = False
        for p in spatial_cfg["multi_phrases"]:
            words = p["phrase"].lower().split()
            if (len(words) >= 2 and
                    i + len(words) - 1 < len(decoded) and
                    decoded[i:i + len(words)] == words):
                out.append((i, p["phrase"], p["canonical"]))
                i += len(words)
                matched = True
                break
        if not matched:
            tok = decoded[i]
            if tok in spatial_cfg["single_lookup"]:
                out.append((i, tok, spatial_cfg["single_lookup"][tok]))
            i += 1
    return out


def find_token_steps(token_labels: list, word: str) -> list:
    w = word.strip().lower()
    return [i for i, t in enumerate(token_labels) if t.strip().lower() == w]


# ---------------------------------------------------------------------------
# TAM norm + logit-lens 
# ---------------------------------------------------------------------------

_NORM_PATHS = (
    "model.model.language_model.norm",          # Qwen2VL / Qwen2.5VL
    "model.model.language_model.model.norm",    # LLaVA (LlavaModel → LlamaModel)
    "model.language_model.model.norm",          # InternVL3-hf
    "model.language_model.norm",
    "language_model.model.norm",
    "model.model.norm",
    "model.norm",
)


def _get_final_norm(model):
    for path in _NORM_PATHS:
        obj = model
        for p in path.split("."):
            obj = getattr(obj, p, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    return None


def _build_logitlens_logits(outputs, model, layer_idx: int, n_layers: int) -> list:
    """Logit-lens projection at layer_idx — exact replica of notebook cell 7."""
    final_norm = _get_final_norm(model)
    n = (n_layers - 1) - layer_idx
    feat_idx = -(n + 1)
    logits = []
    for hs_step in outputs.hidden_states:
        feats = hs_step[feat_idx]
        with torch.no_grad():
            if final_norm is not None:
                feats = final_norm(feats)
            logits.append(model.lm_head(feats))
    return logits


def _num_rounds(outputs, prompt_len: int) -> tuple:
    num_gen = outputs.sequences.shape[1] - prompt_len
    hs_len  = len(outputs.hidden_states)
    has_prefill = (hs_len == num_gen + 1)
    hs_offset   = 1 if has_prefill else 0
    return min(num_gen, max(0, hs_len - hs_offset)), hs_offset


def _decode_token_labels(outputs, prompt_len: int, processor) -> list:
    gen_ids = outputs.sequences[0][prompt_len:].tolist()
    return [
        processor.tokenizer.decode([tid], skip_special_tokens=False).strip()
        for tid in gen_ids
    ]


def _decode_raw_token_labels(outputs, prompt_len: int, processor) -> list:
    gen_ids = outputs.sequences[0][prompt_len:].tolist()
    try:
        return processor.tokenizer.convert_ids_to_tokens(gen_ids)
    except Exception:
        return [processor.tokenizer.decode([tid], skip_special_tokens=False) for tid in gen_ids]


def _clean_piece(tok: str) -> str:
    t = tok.replace("▁", "").replace("Ġ", "")
    if t.startswith("##"):
        t = t[2:]
    return t


def _build_step_word_map(raw_tokens: list, token_labels: list) -> dict:
    """Map each token step to a word group so CSV can aggregate subtokens."""
    if not token_labels:
        return {}

    groups = []
    for i, rtok in enumerate(raw_tokens):
        if i == 0:
            groups.append([i])
            continue

        prev = raw_tokens[i - 1]
        cur_piece = _clean_piece(rtok)
        prev_piece = _clean_piece(prev)

        starts_new = False
        if rtok.startswith(("▁", "Ġ")):
            starts_new = True
        elif rtok.startswith("##"):
            starts_new = False
        elif re.match(r"^[^A-Za-z0-9]+$", cur_piece or ""):
            starts_new = True
        elif re.match(r"^[^A-Za-z0-9]+$", prev_piece or ""):
            starts_new = True
        else:
            # Default: keep contiguous pieces together (e.g., c + rou + ched).
            starts_new = False

        if starts_new:
            groups.append([i])
        else:
            groups[-1].append(i)

    step_map = {}
    for gid, steps in enumerate(groups):
        word = "".join(token_labels[s] for s in steps).strip()
        if not word:
            word = "".join(_clean_piece(raw_tokens[s]) for s in steps).strip()
        for s in steps:
            step_map[s] = {
                "word_id": gid,
                "word": word,
                "word_step_start": steps[0],
                "word_step_end": steps[-1],
                "word_n_subtokens": len(steps),
            }
    return step_map


def aggregate_rows_by_word(rows: list) -> list:
    """Aggregate per-step metric rows into per-word rows."""
    from collections import defaultdict

    buckets = defaultdict(list)
    for r in rows:
        key = (
            r["image"], r["layer"],
            r.get("word_id", r["step"]),
            r.get("word", r.get("token", "")),
            r["target_type"], r["target"],
        )
        buckets[key].append(r)

    out = []
    metric_keys = ["obj_iou", "iou_hard", "io_ratio", "wdp", "func_iou", "f1_iou"]

    for (image, layer, word_id, word, target_type, target), rlist in buckets.items():
        row = {
            "image": image,
            "layer": layer,
            "step": min(r["step"] for r in rlist),
            "step_end": max(r["step"] for r in rlist),
            "token": word,
            "word_id": word_id,
            "word": word,
            "word_n_subtokens": max(r.get("word_n_subtokens", 1) for r in rlist),
            "target_type": target_type,
            "target": target,
        }
        for mk in metric_keys:
            vals = [r[mk] for r in rlist if isinstance(r.get(mk), float) and not math.isnan(r[mk])]
            row[mk] = (sum(vals) / len(vals)) if vals else float("nan")
        out.append(row)

    out.sort(key=lambda r: (r["image"], r["layer"], r["step"], r["target_type"], r["target"]))
    return out


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------

def run_inference(model, processor, image_path: str, prompt: str,
                  model_type: str) -> dict:
    """
    Run generate() and return a context dict with everything needed for
    TAM + evaluation.  Supports qwen2vl, qwen25vl, internvl3, llava.
    """
    from qwen_utils import process_vision_info

    if model_type in ("qwen2vl", "qwen25vl"):
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text",  "text": prompt},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt").to(model.device)
        vision_shape = (
            int(inputs["image_grid_thw"][0, 1]) // 2,
            int(inputs["image_grid_thw"][0, 2]) // 2,
        )
        vis_inputs  = image_inputs
        special_ids = {
            "img_id":    [151652, 151653],
            "prompt_id": [151653, [151645, 198, 151644, 77091]],
            "answer_id": [[198, 151644, 77091, 198], -1],
        }

    elif model_type == "internvl3":
        image = Image.open(image_path).convert("RGB").resize((448, 448))
        conversation = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]}]
        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(text=[text_prompt], images=[image],
                           padding=True, return_tensors="pt").to(model.device).to(model.dtype)
        vision_shape = (16, 16)
        vis_inputs   = image
        special_ids  = {
            "img_id":    [151665, 151666],
            "prompt_id": [[151666, 198], [151645, 198, 151644, 77091]],
            "answer_id": [[198, 151644, 77091, 198], -1],
        }

    elif model_type == "llava":
        image = Image.open(image_path).convert("RGB").resize((336, 336))
        conversation = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]}]
        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs_raw  = processor(text=text_prompt, images=image,
                                return_tensors="pt", padding=True)
        inputs      = {k: v.to(model.device) for k, v in inputs_raw.items()}
        vision_shape = (24, 24)
        vis_inputs   = image
        special_ids  = {
            "img_id":    [32000, 32000],
            "prompt_id": [32000, [319, 1799, 9047, 13566, 29901]],
            "answer_id": [[319, 1799, 9047, 13566, 29901], -1],
        }
        # keep inputs_raw to retrieve prompt_len correctly
        inputs["_input_ids_raw"] = inputs_raw["input_ids"]

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    outputs = model.generate(
        **{k: v for k, v in inputs.items() if not k.startswith("_")},
        max_new_tokens=256, use_cache=True,
        output_hidden_states=True, return_dict_in_generate=True,
    )

    if model_type == "llava":
        prompt_len = inputs["_input_ids_raw"].shape[1]
    else:
        prompt_len = inputs["input_ids"].shape[1]

    generated_ids = outputs.sequences
    num_rounds, hs_offset = _num_rounds(outputs, prompt_len)
    token_labels = _decode_token_labels(outputs, prompt_len, processor)
    raw_token_labels = _decode_raw_token_labels(outputs, prompt_len, processor)
    gen_text = processor.tokenizer.decode(
        generated_ids[0][prompt_len:].tolist(), skip_special_tokens=True
    )

    return {
        "outputs":       outputs,
        "inputs":        inputs,
        "generated_ids": generated_ids,
        "prompt_len":    prompt_len,
        "num_rounds":    num_rounds,
        "hs_offset":     hs_offset,
        "token_labels":  token_labels,
        "raw_token_labels": raw_token_labels,
        "gen_text":      gen_text,
        "vision_shape":  vision_shape,
        "vis_inputs":    vis_inputs,
        "special_ids":   special_ids,
        "n_layers":      len(outputs.hidden_states[0]),
    }


# ---------------------------------------------------------------------------
# TAM grid helper  (per-token across layers)
# ---------------------------------------------------------------------------

def _safe_name(s: str, step: int, maxlen: int = 30) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in s.strip())
    return f"step_{step:04d}_{safe.strip('_')[:maxlen] or 'tok'}"


def _make_layer_grid(layer_paths, token_label, out_path, cols=8,
                     pad=8, label_h=22, bg=(0,0,0), fg=(255,255,255)):
    tiles = []
    for layer_idx, p in layer_paths:
        im = None
        if p and Path(p).exists():
            try:
                im = Image.open(p).convert("RGB")
            except Exception:
                pass
        tiles.append((layer_idx, im))

    valid = [im for _, im in tiles if im is not None]
    if not valid:
        return
    tw = max(im.size[0] for im in valid)
    th = max(im.size[1] for im in valid)
    resized = [(li, im.resize((tw,th), Image.BILINEAR) if im and im.size!=(tw,th) else im)
               for li, im in tiles]

    rows = math.ceil(len(resized) / cols)
    W = cols*tw + (cols+1)*pad
    H = rows*(th+label_h) + (rows+1)*pad
    canvas = Image.new("RGB", (W, H), bg)
    draw   = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for i, (li, im) in enumerate(resized):
        r, c = i // cols, i % cols
        x0 = pad + c*(tw+pad); y0 = pad + r*(th+label_h+pad)
        draw.rectangle([x0,y0,x0+tw,y0+label_h], fill=bg)
        draw.text((x0+4, y0+4), f"L{li}", fill=fg, font=font)
        if im:
            canvas.paste(im, (x0, y0+label_h))
        else:
            draw.rectangle([x0,y0+label_h,x0+tw,y0+label_h+th], outline=fg)

    header_h = 30
    out_img = Image.new("RGB", (W, H+header_h), bg)
    out_img.paste(canvas, (0, header_h))
    draw2 = ImageDraw.Draw(out_img)
    draw2.text((pad, 7), f"token: {token_label[:70]}", fill=fg, font=font)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out_img.save(str(out_path), quality=95)


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

def evaluate_image(ctx: dict, obj_masks: dict, spatial_cfg: dict,
                   model, logits_last: list, layer_logits: dict,
                   run_layers: list, vis_dir: Path, grids_dir: Path,
                   stem: str, image_queries: list | None = None) -> list:
    """
    Run TAM for all steps (and all layers if layer_logits is given),
    compute metrics against obj_masks and spatial relation masks.

    Returns list of row dicts ready for CSV.
    """
    from tam_words import TAM

    generated_ids = ctx["generated_ids"]
    vision_shape  = ctx["vision_shape"]
    special_ids   = ctx["special_ids"]
    vis_inputs    = ctx["vis_inputs"]
    token_labels  = ctx["token_labels"]
    step_word_map = ctx.get("step_word_map", {})
    num_rounds    = ctx["num_rounds"]
    processor     = ctx["processor"]

    rows = []

    # layer_step_paths[layer_idx][step] = Path  — for grid building
    layer_step_paths: dict[int, dict[int, Path]] = {li: {} for li in run_layers}

    for layer_idx in run_layers:
        logits = layer_logits.get(layer_idx, logits_last)
        layer_dir = vis_dir / stem / f"layer_{layer_idx:03d}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        queries = image_queries or []

        def run_query(q: dict):
            obj_name = q["object"]
            target_word = q["word"]
            kind = q["kind"]
            mask_id = str(q.get("mask_id", ""))
            if mask_id not in obj_masks:
                return None

            target_step, matched_labels = _find_target_final_step(step_word_map, target_word, token_labels)
            if target_step < 0 or target_step >= num_rounds:
                return None

            preserve_words = [obj_name]
            preserve_words.extend(matched_labels[:-1])
            preserve_words = list(dict.fromkeys(w for w in preserve_words if w))

            local_img_scores = []
            img_map_local = None
            for step in range(target_step + 1):
                save_path = layer_dir / f"{kind}_{obj_name}_{target_word}_step_{step:04d}.jpg"
                img_map_local = TAM(
                    generated_ids[0].cpu().tolist(),
                    vision_shape, logits, special_ids,
                    vis_inputs, processor,
                    str(save_path) if step == target_step else "",
                    step,
                    local_img_scores,
                    False,
                    preserve_prev_words=preserve_words,
                )
                layer_step_paths[layer_idx][step] = save_path

            if img_map_local is None:
                return None

            m = compute_all_metrics(img_map_local, obj_masks[mask_id])
            tok_lbl = token_labels[target_step] if target_step < len(token_labels) else ""
            wmeta = step_word_map.get(target_step, {
                "word_id": target_step,
                "word": tok_lbl,
                "word_step_start": target_step,
                "word_step_end": target_step,
                "word_n_subtokens": 1,
            })
            return {
                "image": stem, "layer": layer_idx, "step": target_step,
                "token": tok_lbl,
                "word_id": wmeta["word_id"],
                "word": wmeta["word"],
                "word_step_start": wmeta["word_step_start"],
                "word_step_end": wmeta["word_step_end"],
                "word_n_subtokens": wmeta["word_n_subtokens"],
                "target_type": f"query_{kind}",
                "target": obj_name,
                "query_object": obj_name,
                "query_word": target_word,
                "query_mask": mask_id,
                "query_pair": f"{target_word}+{obj_name}",
                **m,
            }

        if queries:
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
                futs = [ex.submit(run_query, q) for q in queries]
                for f in concurrent.futures.as_completed(futs):
                    row = f.result()
                    if row is not None:
                        rows.append(row)

    # Build per-token grids (token across layers)
    if grids_dir is not None and len(run_layers) > 1:
        for step in range(num_rounds):
            tok_lbl = token_labels[step] if step < len(token_labels) else f"step{step}"
            out_path = grids_dir / stem / _safe_name(tok_lbl, step) / "grid_layers.jpg"
            layer_paths = [(li, layer_step_paths[li].get(step)) for li in run_layers]
            _make_layer_grid(layer_paths, tok_lbl, out_path)

    return rows


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_model(model_name: str) -> tuple:
    """Returns (model, processor, model_type)."""
    name_lower = model_name.lower()

    if "qwen2.5" in name_lower or "qwen2_5" in name_lower:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_name)
        return model, processor, "qwen25vl"

    elif "qwen2" in name_lower:
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_name)
        return model, processor, "qwen2vl"

    elif "internvl" in name_lower:
        from transformers import AutoModelForImageTextToText, AutoProcessor
        model = AutoModelForImageTextToText.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_name)
        return model, processor, "internvl3"

    elif "llava" in name_lower:
        from transformers import LlavaForConditionalGeneration, AutoProcessor
        model = LlavaForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )
        model.eval()
        processor = AutoProcessor.from_pretrained(model_name)
        return model, processor, "llava"

    else:
        raise ValueError(f"Unrecognised model: {model_name}. "
                         "Supported: Qwen2-VL, Qwen2.5-VL, InternVL3, LLaVA.")


def _check_norm(model):
    norm = _get_final_norm(model)
    if norm is None:
        print("[WARN] final norm not found — logit-lens heatmaps will be noisy")
    else:
        print(f"[OK]   final norm: {type(norm).__name__}")


# ---------------------------------------------------------------------------
# Config — edit these variables before running
# ---------------------------------------------------------------------------

MODEL_NAME  = "Qwen/Qwen2-VL-2B-Instruct"
IMAGES_DIR  = Path(r"/content/FVAB-TAM-Gruppo16/tam-logit-lenses/ll_tam/data/coco2014/image")          # directory with test images
MASKS_DIR   = Path(r"/content/FVAB-TAM-Gruppo16/Fase_3/merged_masks")              # merged masks/<stem>/<1.png, 2.png, 3.png>
CONFIG_PATH = Path(r"/content/FVAB-TAM-Gruppo16/tam-logit-lenses/ll_tam/rel_config.yaml")
OUT_DIR     = Path(r"/content/FVAB-TAM-Gruppo16/tam-logit-lenses/ll_tam/results")
VIS_DIR     = Path(r"/content/FVAB-TAM-Gruppo16/tam-logit-lenses/ll_tam/vis_results")        # TAM heatmaps per layer/step
GRIDS_DIR   = Path(r"/content/FVAB-TAM-Gruppo16/tam-logit-lenses/ll_tam/token_grids")        # per-token cross-layer grids
QUERY_TXT   = Path(r"/content/FVAB-TAM-Gruppo16/Fase_3/target_img_ids.txt") # blocks: image_id + up to 3 relation lines

PROMPT      = "Describe this image in two sentences."
MAX_NEW_TOKENS = 256

# Layer selection:
#   None      → last layer only (standard TAM, fast)
#   "all"     → all layers (logit-lens, slow)
LAYERS = None
AGGREGATE_SUBTOKENS = True  # If True, CSV rows are aggregated per reconstructed word.

# Subset of images to process (list of filenames, or None for all)
IMAGES_LIST = None   # e.g. ["img1.jpg", "img2.jpg"]

# Debug mode: set to True to enable detailed logging for target word matching
DEBUG_MATCHING = False

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from collections import defaultdict

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    grids_dir = GRIDS_DIR if LAYERS is not None or LAYERS == "all" else None
    # always create grids dir when running multi-layer
    if LAYERS == "all" or (isinstance(LAYERS, list) and len(LAYERS) > 1):
        grids_dir = GRIDS_DIR
        grids_dir.mkdir(parents=True, exist_ok=True)

    # Collect images
    img_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    if IMAGES_LIST:
        image_files = [IMAGES_DIR / fn for fn in IMAGES_LIST
                       if (IMAGES_DIR / fn).suffix.lower() in img_extensions]
    else:
        image_files = sorted(p for p in IMAGES_DIR.iterdir()
                             if p.is_file() and p.suffix.lower() in img_extensions)

    if not image_files:
        sys.exit(f"No images found in {IMAGES_DIR}")

    print(f"Images to evaluate: {len(image_files)}")

    spatial_cfg = load_spatial_config(str(CONFIG_PATH))
    print(f"Spatial config: {len(spatial_cfg['prepositions'])} prepositions loaded")
    query_map = load_object_word_queries(QUERY_TXT)
    print(f"Query txt path: {QUERY_TXT}")
    print(f"Query blocks loaded: {len(query_map)}")

    model, processor, model_type = load_model(MODEL_NAME)
    model.eval()
    _check_norm(model)
    print(f"Model type: {model_type}")

    all_rows = []

    for img_path in image_files:
        stem = img_path.stem
        print(f"\n--- {stem} ---")

        canon_stem = _canonical_image_id(stem)
        image_queries = query_map.get(canon_stem, [])
        if not image_queries:
            print(f"  [SKIP] no query block for this image (query key: {canon_stem})")
            continue

        obj_masks = get_object_masks(stem, MASKS_DIR)
        if not obj_masks:
            print(f"  [SKIP] no masks found in {MASKS_DIR / stem}")
            continue
        print(f"  masks: {list(obj_masks.keys())}")

        ctx = run_inference(model, processor, str(img_path), PROMPT, model_type)
        ctx["processor"] = processor
        ctx["step_word_map"] = _build_step_word_map(
            ctx.get("raw_token_labels", []),
            ctx.get("token_labels", []),
        )
        print(f"  generated: '{ctx['gen_text']}'  ({ctx['num_rounds']} steps)")
        print(f"  raw_token_labels: {ctx.get('raw_token_labels', [])}")
        print(f"  token_labels: {ctx.get('token_labels', [])}")
        print(f"  step_word_map: {ctx.get('step_word_map', {})}")

        n_layers = ctx["n_layers"]
        outputs  = ctx["outputs"]

        if LAYERS == "all":
            run_layers = list(range(n_layers))
        elif isinstance(LAYERS, list):
            run_layers = LAYERS
        else:
            run_layers = [n_layers - 1]   # last layer only

        layer_logits: dict = {}
        for li in run_layers:
            layer_logits[li] = _build_logitlens_logits(outputs, model, li, n_layers)

        rows = evaluate_image(
            ctx=ctx,
            obj_masks=obj_masks,
            spatial_cfg=spatial_cfg,
            model=model,
            logits_last=layer_logits[run_layers[-1]],
            layer_logits=layer_logits,
            run_layers=run_layers,
            vis_dir=VIS_DIR,
            grids_dir=grids_dir,
            stem=stem,
            image_queries=image_queries,
        )
        all_rows.extend(rows)
        print(f"  metric rows produced: {len(rows)}")

    if not all_rows:
        print("\nNo rows produced — check masks and image names.")
        sys.exit(0)

    if AGGREGATE_SUBTOKENS:
        all_rows = aggregate_rows_by_word(all_rows)
        print(f"Aggregated per-word rows: {len(all_rows)}")

    # Full CSV
    csv_path = OUT_DIR / "results.csv"
    fieldnames = ["image", "layer", "step", "step_end", "token",
                  "word_id", "word", "word_n_subtokens",
                  "query_object", "query_word", "query_pair",
                  "target_type", "target",
                  "obj_iou", "iou_hard", "io_ratio", "wdp",
                  "func_iou", "f1_iou"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nFull CSV -> {csv_path}  ({len(all_rows)} rows)")

    # Summary per (target_type, target, layer)
    agg: dict = defaultdict(list)
    for r in all_rows:
        key = (r["target_type"], r["target"], r["layer"])
        agg[key].append(r)

    summary_rows = []
    for (ttype, target, layer), rlist in sorted(agg.items()):
        def avg(metric):
            vals = [r[metric] for r in rlist
                    if isinstance(r[metric], float) and not math.isnan(r[metric])]
            return sum(vals) / len(vals) if vals else float("nan")
        
        obj  = avg("obj_iou")
        hard = avg("iou_hard")
        io   = avg("io_ratio")
        wdp  = avg("wdp")
        func = avg("func_iou")
        
        f1   = float("nan")
        if not math.isnan(obj) and not math.isnan(func) and (obj + func) > 0:
            f1 = 2 * obj * func / (obj + func)
            
        summary_rows.append({
            "target_type": ttype, "target": target, "layer": layer,
            "n": len(rlist),
            "obj_iou":  round(obj,  4), "iou_hard": round(hard, 4),
            "io_ratio": round(io,   4), "wdp":      round(wdp,  4),
            "func_iou": round(func, 4),
            "f1_iou":   round(f1,   4) if not math.isnan(f1) else "nan",
        })

    summary_path = OUT_DIR / "summary.csv"
    sum_fields = ["target_type", "target", "layer", "n",
                  "obj_iou", "iou_hard", "io_ratio", "wdp", "func_iou", "f1_iou"]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sum_fields)
        w.writeheader()
        w.writerows(summary_rows)
    print(f"Summary CSV -> {summary_path}")

    print("\n" + "=" * 82)
    print(f"{'target_type':<22} {'target':<20} {'L':>3} {'obj_iou':>8} "
          f"{'iou_hard':>8} {'io_ratio':>8} {'wdp':>7} {'func_iou':>8} {'f1_iou':>7}")
    print("-" * 82)
    for r in summary_rows:
        print(f"{r['target_type']:<22} {r['target']:<20} {r['layer']:>3} "
              f"{r['obj_iou']:>8} {r['iou_hard']:>8} {r['io_ratio']:>8} "
              f"{r['wdp']:>7} {r['func_iou']:>8} {r['f1_iou']:>7}")
    print("=" * 82)
