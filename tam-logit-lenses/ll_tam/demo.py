import os
import math
import torch
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    AutoModelForImageTextToText,
    LlavaForConditionalGeneration,
)
from qwen_utils import process_vision_info
from PIL import Image, ImageDraw, ImageFont
from tam import TAM
from pathlib import Path


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

# Correct norm paths per architecture (checked against transformers source):
#   Qwen2VL:   model.model.language_model.norm
#   Qwen2.5VL: model.model.language_model.norm  (same structure)
#   InternVL3: model.language_model.model.norm   (AutoModelForImageTextToText)
#   LLaVA:     model.language_model.model.norm   (or model.model.norm)
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


def _build_logitlens_logits(outputs, model, layer_idx: int, n_layers: int):
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


def _image_stem(image_path) -> str:
    return Path(image_path).stem


def _decode_tokens(outputs, prompt_len: int, processor) -> list:
    gen_ids = outputs.sequences[0][prompt_len:].tolist()
    return [
        processor.tokenizer.decode([tid], skip_special_tokens=False).strip()
        for tid in gen_ids
    ]


def _num_rounds(outputs, prompt_len: int) -> tuple:
    """Returns (num_rounds, hs_offset) matching notebook cell 6 logic."""
    num_gen_steps = outputs.sequences.shape[1] - prompt_len
    hs_len = len(outputs.hidden_states)
    has_prefill = (hs_len == num_gen_steps + 1)
    hs_offset = 1 if has_prefill else 0
    num_rounds = min(num_gen_steps, max(0, hs_len - hs_offset))
    return num_rounds, hs_offset


# ---------------------------------------------------------------------------
# Grid: per-token across layers
#
# Output: grids_dir/<img_stem>/step_XXXX_<token>/grid_layers.jpg
# Each grid shows the same token's heatmap across all requested layers,
# with "L{layer_idx}" label on each tile.
# ---------------------------------------------------------------------------

def _make_layer_grid_for_token(
    layer_heatmap_paths,   # list of (layer_idx, Path or None)
    token_label: str,
    out_path: Path,
    cols: int = 8,
    pad: int = 8,
    label_h: int = 22,
    bg=(0, 0, 0),
    fg=(255, 255, 255),
):
    tiles = []
    for layer_idx, p in layer_heatmap_paths:
        im = None
        if p is not None and p.exists():
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

    resized = []
    for layer_idx, im in tiles:
        if im is not None and im.size != (tw, th):
            im = im.resize((tw, th), Image.BILINEAR)
        resized.append((layer_idx, im))

    n = len(resized)
    rows = math.ceil(n / cols)
    W = cols * tw + (cols + 1) * pad
    H = rows * (th + label_h) + (rows + 1) * pad

    canvas = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for i, (layer_idx, im) in enumerate(resized):
        r, c = i // cols, i % cols
        x0 = pad + c * (tw + pad)
        y0 = pad + r * (th + label_h + pad)
        draw.rectangle([x0, y0, x0 + tw, y0 + label_h], fill=bg)
        draw.text((x0 + 4, y0 + 4), f"L{layer_idx}", fill=fg, font=font)
        if im is not None:
            canvas.paste(im, (x0, y0 + label_h))
        else:
            draw.rectangle([x0, y0 + label_h, x0 + tw, y0 + label_h + th], outline=fg)

    # header with token name
    header_h = 30
    out_img = Image.new("RGB", (W, H + header_h), bg)
    out_img.paste(canvas, (0, header_h))
    draw2 = ImageDraw.Draw(out_img)
    draw2.text((pad, 7), f"token: {token_label[:70]}", fill=fg, font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(str(out_path), quality=95)


def _safe_folder_name(token_str: str, step_idx: int, maxlen: int = 30) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in token_str.strip())
    safe = safe.strip("_")[:maxlen] or "tok"
    return f"step_{step_idx:04d}_{safe}"


def _build_per_token_grids(stem, token_labels, num_rounds, run_layers,
                            layer_step_paths, grids_root, cols=8):
    """
    For each generated token, build a grid across all layers.
    Saves to: grids_root/<stem>/step_XXXX_<token>/grid_layers.jpg
    layer_step_paths: {layer_idx: {step_idx: Path}}
    """
    img_grids_dir = grids_root / stem
    img_grids_dir.mkdir(parents=True, exist_ok=True)

    for step_idx in range(num_rounds):
        tok_lbl = token_labels[step_idx] if step_idx < len(token_labels) else f"step{step_idx}"
        out_path = img_grids_dir / _safe_folder_name(tok_lbl, step_idx) / "grid_layers.jpg"
        layer_paths = [
            (layer_idx, layer_step_paths.get(layer_idx, {}).get(step_idx))
            for layer_idx in run_layers
        ]
        _make_layer_grid_for_token(layer_paths, tok_lbl, out_path, cols=cols)
        print(f"  grid [{step_idx}] '{tok_lbl}' -> {out_path}")


# ---------------------------------------------------------------------------
# Demo functions
# ---------------------------------------------------------------------------

def tam_demo_for_qwen2_vl(
    image_path,
    prompt_text: str,
    save_dir: str = "vis_results",
    grids_dir: str = "grids",
    all_layers: bool = False,
    layers: list = None,
    grid_cols: int = 8,
):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(grids_dir, exist_ok=True)

    model_name = "Qwen/Qwen2-VL-2B-Instruct"
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype="auto", device_map="cpu"
    )
    processor = AutoProcessor.from_pretrained(model_name)

    # verify norm is found (warn if not, heatmaps will be noisy)
    norm = _get_final_norm(model)
    if norm is None:
        print("[WARN] final norm not found — logit-lens heatmaps will be noisy")
    else:
        print(f"[OK] final norm: {type(norm).__name__}")

    if isinstance(image_path, list):
        messages = [{"role": "user", "content": [
            {"type": "video", "video": image_path},
            {"type": "text", "text": prompt_text},
        ]}]
    else:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt_text},
        ]}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(
        **inputs, max_new_tokens=256, use_cache=True,
        output_hidden_states=True, return_dict_in_generate=True,
    )

    # tensor 2D — TAM receives [0].cpu().tolist() internally
    generated_ids = outputs.sequences

    special_ids = {
        'img_id':    [151652, 151653],
        'prompt_id': [151653, [151645, 198, 151644, 77091]],
        'answer_id': [[198, 151644, 77091, 198], -1],
    }

    if isinstance(image_path, list):
        vision_shape = (
            int(inputs["video_grid_thw"][0, 0]),
            int(inputs["video_grid_thw"][0, 1]) // 2,
            int(inputs["video_grid_thw"][0, 2]) // 2,
        )
        vis_inputs = [[video_inputs[0][i] for i in range(len(video_inputs[0]))]]
        stem = "video"
    else:
        vision_shape = (
            int(inputs["image_grid_thw"][0, 1]) // 2,
            int(inputs["image_grid_thw"][0, 2]) // 2,
        )
        vis_inputs = image_inputs
        stem = _image_stem(image_path)

    n_layers   = len(outputs.hidden_states[0])
    prompt_len = inputs["input_ids"].shape[1]
    num_rounds, hs_offset = _num_rounds(outputs, prompt_len)
    token_labels = _decode_tokens(outputs, prompt_len, processor)

    gen_text = processor.tokenizer.decode(
        generated_ids[0][prompt_len:].tolist(), skip_special_tokens=True
    )
    print(f"[{stem}] steps={num_rounds} | '{gen_text}'")

    img_dir = Path(save_dir) / stem
    img_dir.mkdir(parents=True, exist_ok=True)
    run_layers = layers if layers is not None else list(range(n_layers))

    if not all_layers:
        logits = _build_logitlens_logits(outputs, model, n_layers - 1, n_layers)
        raw_map_records = []
        for i in range(num_rounds):
            save_path = img_dir / f"step_{i:04d}.jpg"
            TAM(generated_ids[0].cpu().tolist(), vision_shape, logits, special_ids,
                vis_inputs, processor, str(save_path), i, raw_map_records, False)
    else:
        layer_step_paths = {}
        for layer_idx in run_layers:
            layer_dir = img_dir / f"layer_{layer_idx:03d}"
            layer_dir.mkdir(parents=True, exist_ok=True)
            logits = _build_logitlens_logits(outputs, model, layer_idx, n_layers)
            img_scores_list = []
            layer_step_paths[layer_idx] = {}
            for round_idx in range(num_rounds):
                save_path = layer_dir / f"step_{round_idx:04d}.jpg"
                TAM(generated_ids[0].cpu().tolist(), vision_shape, logits, special_ids,
                    vis_inputs, processor, str(save_path), round_idx, img_scores_list, False)
                layer_step_paths[layer_idx][round_idx] = save_path

        _build_per_token_grids(stem, token_labels, num_rounds, run_layers,
                               layer_step_paths, Path(grids_dir), grid_cols)


def tam_demo_for_internvl3(
    image_path: str,
    prompt_text: str,
    save_dir: str = "vis_results",
    grids_dir: str = "grids",
    all_layers: bool = False,
    layers: list = None,
    grid_cols: int = 8,
):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(grids_dir, exist_ok=True)

    model_checkpoint = "OpenGVLab/InternVL3-1B-hf"
    processor = AutoProcessor.from_pretrained(model_checkpoint)
    model = AutoModelForImageTextToText.from_pretrained(
        model_checkpoint, device_map="cuda", torch_dtype=torch.bfloat16
    )

    norm = _get_final_norm(model)
    if norm is None:
        print("[WARN] final norm not found")
    else:
        print(f"[OK] final norm: {type(norm).__name__}")

    image = Image.open(image_path).convert("RGB").resize((448, 448))
    conversation = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": prompt_text},
    ]}]
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(
        text=[text_prompt], images=[image], padding=True, return_tensors="pt",
    ).to(model.device).to(model.dtype)

    outputs = model.generate(
        **inputs, max_new_tokens=256,
        output_hidden_states=True, return_dict_in_generate=True,
    )

    generated_ids = outputs.sequences
    special_ids = {
        'img_id':    [151665, 151666],
        'prompt_id': [[151666, 198], [151645, 198, 151644, 77091]],
        'answer_id': [[198, 151644, 77091, 198], -1],
    }
    vision_shape = (16, 16)
    vis_inputs   = image
    stem         = _image_stem(image_path)
    n_layers     = len(outputs.hidden_states[0])
    prompt_len   = inputs["input_ids"].shape[1]
    num_rounds, hs_offset = _num_rounds(outputs, prompt_len)
    token_labels = _decode_tokens(outputs, prompt_len, processor)

    img_dir = Path(save_dir) / stem
    img_dir.mkdir(parents=True, exist_ok=True)
    run_layers = layers if layers is not None else list(range(n_layers))

    if not all_layers:
        logits = _build_logitlens_logits(outputs, model, n_layers - 1, n_layers)
        raw_map_records = []
        for i in range(num_rounds):
            save_path = img_dir / f"step_{i:04d}.jpg"
            TAM(generated_ids[0].cpu().tolist(), vision_shape, logits, special_ids,
                vis_inputs, processor, str(save_path), i, raw_map_records, False)
    else:
        layer_step_paths = {}
        for layer_idx in run_layers:
            layer_dir = img_dir / f"layer_{layer_idx:03d}"
            layer_dir.mkdir(parents=True, exist_ok=True)
            logits = _build_logitlens_logits(outputs, model, layer_idx, n_layers)
            img_scores_list = []
            layer_step_paths[layer_idx] = {}
            for round_idx in range(num_rounds):
                save_path = layer_dir / f"step_{round_idx:04d}.jpg"
                TAM(generated_ids[0].cpu().tolist(), vision_shape, logits, special_ids,
                    vis_inputs, processor, str(save_path), round_idx, img_scores_list, False)
                layer_step_paths[layer_idx][round_idx] = save_path
        _build_per_token_grids(stem, token_labels, num_rounds, run_layers,
                               layer_step_paths, Path(grids_dir), grid_cols)


def tam_demo_for_llava(
    image_path: str,
    prompt_text: str,
    save_dir: str = "vis_results",
    grids_dir: str = "grids",
    all_layers: bool = False,
    layers: list = None,
    model_name: str = "llava-hf/llava-1.5-7b-hf",
    grid_cols: int = 8,
):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(grids_dir, exist_ok=True)

    processor = AutoProcessor.from_pretrained(model_name)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    norm = _get_final_norm(model)
    if norm is None:
        print("[WARN] final norm not found")
    else:
        print(f"[OK] final norm: {type(norm).__name__}")

    conversation = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": prompt_text},
    ]}]
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    image = Image.open(image_path).convert("RGB").resize((336, 336))
    inputs_raw = processor(text=text_prompt, images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs_raw.items()}

    outputs = model.generate(
        **inputs, max_new_tokens=256, use_cache=True,
        output_hidden_states=True, return_dict_in_generate=True,
    )

    generated_ids = outputs.sequences
    special_ids = {
        'img_id':    [32000, 32000],
        'prompt_id': [32000, [319, 1799, 9047, 13566, 29901]],
        'answer_id': [[319, 1799, 9047, 13566, 29901], -1],
    }
    vision_shape = (24, 24)
    vis_inputs   = image
    stem         = _image_stem(image_path)
    n_layers     = len(outputs.hidden_states[0])
    prompt_len   = inputs_raw["input_ids"].shape[1]
    num_rounds, hs_offset = _num_rounds(outputs, prompt_len)
    token_labels = _decode_tokens(outputs, prompt_len, processor)

    img_dir = Path(save_dir) / stem
    img_dir.mkdir(parents=True, exist_ok=True)
    run_layers = layers if layers is not None else list(range(n_layers))

    if not all_layers:
        logits = _build_logitlens_logits(outputs, model, n_layers - 1, n_layers)
        raw_map_records = []
        for i in range(num_rounds):
            save_path = img_dir / f"step_{i:04d}.jpg"
            TAM(generated_ids[0].cpu().tolist(), vision_shape, logits, special_ids,
                vis_inputs, processor, str(save_path), i, raw_map_records, False)
    else:
        layer_step_paths = {}
        for layer_idx in run_layers:
            layer_dir = img_dir / f"layer_{layer_idx:03d}"
            layer_dir.mkdir(parents=True, exist_ok=True)
            logits = _build_logitlens_logits(outputs, model, layer_idx, n_layers)
            img_scores_list = []
            layer_step_paths[layer_idx] = {}
            for round_idx in range(num_rounds):
                save_path = layer_dir / f"step_{round_idx:04d}.jpg"
                TAM(generated_ids[0].cpu().tolist(), vision_shape, logits, special_ids,
                    vis_inputs, processor, str(save_path), round_idx, img_scores_list, False)
                layer_step_paths[layer_idx][round_idx] = save_path
        _build_per_token_grids(stem, token_labels, num_rounds, run_layers,
                               layer_step_paths, Path(grids_dir), grid_cols)

if __name__ == "__main__":

    '''
    Demo code for original TAM.
    # single img demo (qwen)
    img = "imgs/demo.jpg"
    prompt = "Describe this image."
    tam_demo_for_qwen2_vl(img, prompt, save_dir='imgs/vis_img')

    # single img demo (internvl)
    tam_demo_for_internvl3(img, prompt, save_dir='imgs/vis_img_internvl')

    # video demo (qwen)
    imgs = []
    for i in range(10):
        # QWen merges next frames, repeating to vis each frame
        imgs.extend(["imgs/frames/%s.jpg" % (str(i).zfill(4))] * 2)
    prompt = "Describe this video."
    tam_demo_for_qwen2_vl(imgs, prompt, save_dir='imgs/vis_video')
    '''
    img_path = Path("test")
    prompt = "Describe this image in a phrase."
    #img    = Path("imgs/demo.jpg")
    for img in img_path.iterdir():
        tam_demo_for_qwen2_vl(str(img), prompt, save_dir="imgs/ll_qwen",grids_dir="imgs/ll_qwen_token_grids",  all_layers=True)
        #tam_demo_for_internvl3(str(img), prompt, save_dir="imgs/ll_internvl", all_layers=True)
        #tam_demo_for_llava(str(img),     prompt, save_dir="imgs/ll_llava",    all_layers=True)