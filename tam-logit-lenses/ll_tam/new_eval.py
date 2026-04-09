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
    """Returns {obj_name: np.ndarray} for all masks found for this image."""
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
                   stem: str) -> list:
    """
    Run TAM for all steps (and all layers if layer_logits is given),
    compute metrics against obj_masks and spatial relation masks.

    Returns list of row dicts ready for CSV.
    """
    from tam import TAM

    generated_ids = ctx["generated_ids"]
    vision_shape  = ctx["vision_shape"]
    special_ids   = ctx["special_ids"]
    vis_inputs    = ctx["vis_inputs"]
    token_labels  = ctx["token_labels"]
    num_rounds    = ctx["num_rounds"]
    processor     = ctx["processor"]

    rows = []

    # Detect spatial steps
    spatial_steps = find_spatial_steps(token_labels, spatial_cfg)
    spatial_lookup = {step: (phrase, canon)
                      for step, phrase, canon in spatial_steps}

    # layer_step_paths[layer_idx][step] = Path  — for grid building
    layer_step_paths: dict[int, dict[int, Path]] = {li: {} for li in run_layers}

    for layer_idx in run_layers:
        logits = layer_logits.get(layer_idx, logits_last)
        layer_dir = vis_dir / stem / f"layer_{layer_idx:03d}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        img_scores_list = []

        for step in range(num_rounds):
            save_path = layer_dir / f"step_{step:04d}.jpg"
            img_map = TAM(
                generated_ids[0].cpu().tolist(),
                vision_shape, logits, special_ids,
                vis_inputs, processor,
                str(save_path), step, img_scores_list, False,
            )
            layer_step_paths[layer_idx][step] = save_path

            if img_map is None:
                continue

            tok_lbl = token_labels[step] if step < len(token_labels) else ""

            # --- metrics against each object mask ---
            for obj_name, obj_mask in obj_masks.items():
                m = compute_all_metrics(img_map, obj_mask)
                rows.append({
                    "image": stem, "layer": layer_idx, "step": step,
                    "token": tok_lbl,
                    "target_type": "object", "target": obj_name,
                    **m
                })

            # --- metrics against spatial relation mask ---
            if step in spatial_lookup:
                phrase, canon = spatial_lookup[step]
                prep_data = spatial_cfg["prepositions"].get(canon, {})
                # try to find sub/obj from mask names heuristically
                # (use all pairs of available masks)
                mask_names = list(obj_masks.keys())
                for i_s, sub_name in enumerate(mask_names):
                    for obj_name in mask_names:
                        if sub_name == obj_name:
                            continue
                        sub_m = obj_masks[sub_name]
                        obj_m = obj_masks[obj_name]
                        h, w  = sub_m.shape
                        rel_mask = relation_region_mask(
                            canon, sub_m, obj_m,
                            spatial_cfg["prepositions"]
                        )
                        m = compute_all_metrics(img_map, rel_mask)
                        rows.append({
                            "image": stem, "layer": layer_idx, "step": step,
                            "token": tok_lbl,
                            "target_type": f"relation_{canon}",
                            "target": f"{sub_name}__{obj_name}",
                            **m
                        })

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
IMAGES_DIR  = Path("test")          # directory with test images
MASKS_DIR   = Path("masks")              # masks/<stem>/<obj>.png
CONFIG_PATH = Path("rel_config.yaml")
OUT_DIR     = Path("results")
VIS_DIR     = Path("vis_results")        # TAM heatmaps per layer/step
GRIDS_DIR   = Path("token_grids")        # per-token cross-layer grids

PROMPT      = "Describe this image in detail."
MAX_NEW_TOKENS = 256

# Layer selection:
#   None      → last layer only (standard TAM, fast)
#   "all"     → all layers (logit-lens, slow)
#   [0,8,16]  → specific indices
LAYERS = None

# Subset of images to process (list of filenames, or None for all)
IMAGES_LIST = None   # e.g. ["img1.jpg", "img2.jpg"]

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

    model, processor, model_type = load_model(MODEL_NAME)
    model.eval()
    _check_norm(model)
    print(f"Model type: {model_type}")

    all_rows = []

    for img_path in image_files:
        stem = img_path.stem
        print(f"\n--- {stem} ---")

        obj_masks = get_object_masks(stem, MASKS_DIR)
        if not obj_masks:
            print(f"  [SKIP] no masks found in {MASKS_DIR / stem}")
            continue
        print(f"  masks: {list(obj_masks.keys())}")

        ctx = run_inference(model, processor, str(img_path), PROMPT, model_type)
        ctx["processor"] = processor
        print(f"  generated: '{ctx['gen_text']}'  ({ctx['num_rounds']} steps)")

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
        )
        all_rows.extend(rows)
        print(f"  metric rows produced: {len(rows)}")

    if not all_rows:
        print("\nNo rows produced — check masks and image names.")
        sys.exit(0)

    # Full CSV
    csv_path = OUT_DIR / "results.csv"
    fieldnames = ["image", "layer", "step", "token",
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