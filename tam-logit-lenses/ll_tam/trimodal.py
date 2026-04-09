import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Patch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ===========================================================================
# Utilities
# ===========================================================================

# Correct norm paths per architecture to extract raw hidden states correctly
_NORM_PATHS = (
    "model.model.language_model.norm",          
    "model.model.language_model.model.norm",    
    "model.language_model.model.norm",          
    "model.language_model.norm",
    "language_model.model.norm",
    "model.model.norm",
    "model.norm",
)

def _get_final_norm(model):
    """
    Finds and returns the final normalization layer of the LLM.
    This is required to properly project hidden states into the logit space.
    """
    for path in _NORM_PATHS:
        obj = model
        for p in path.split("."):
            obj = getattr(obj, p, None)
            if obj is None: break
        if obj is not None: return obj
    return None

def plot_trimodal_matrix(mat_v, mat_p, mat_h, save_path, token_labels, title="Trimodal Heatmap"):
    """
    Creates an RGB matrix where each cell is a mix of Vision, Prompt, and History weights.
    Red   = Vision (Image Tokens)
    Green = Prompt (Instructions)
    Blue  = History (Autoregressive generated tokens)
    """
    n_steps, n_layers = mat_v.shape
    
    # Mix channels into an RGB image
    rgb = np.stack([mat_v, mat_p, mat_h], axis=-1)
    rgb = np.clip(rgb, 0, 1)

    # Dynamic figure size based on the number of layers and generated tokens
    fig, ax = plt.subplots(figsize=(max(8, n_layers*0.2), max(4, n_steps*0.3)))
    ax.imshow(rgb, aspect='auto', interpolation='nearest')

    # Axes and Labels
    ax.set_xticks(range(0, n_layers, max(1, n_layers//10)))
    ax.set_xticklabels([f"L{i}" for i in range(0, n_layers, max(1, n_layers//10))])

    ax.set_yticks(range(n_steps))
    clean_labels = [lbl.replace('\n', ' ').strip() for lbl in token_labels]
    ax.set_yticklabels(clean_labels, fontsize=8)
    
    ax.set_xlabel("Layers")
    ax.set_title(title, pad=20)

    # Custom Legend
    legend_elements = [
        Patch(facecolor='red', label='Vision (Image Tokens)'),
        Patch(facecolor='green', label='Prompt (Instructions)'),
        Patch(facecolor='blue', label='History (Past Generation)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.01, 1))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# ===========================================================================
# Core Processing Function
# ===========================================================================

def analyze_single_image(model, processor, norm, image_path, prompt_text, out_dir):
    """
    Runs the inference and computes both Attention and Logit Lens trimodal maps
    for a single image.
    """
    stem = Path(image_path).stem
    lm_head = model.lm_head

    # 1. Input Preparation
    messages = [{"role": "user", "content": [
        {"type": "image", "image": str(image_path)},
        {"type": "text", "text": prompt_text},
    ]}]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)

    # 2. Find boundaries for visual and prompt tokens
    input_ids = inputs["input_ids"][0].tolist()
    img_start = input_ids.index(151652) + 1  # Qwen vision start token
    img_end = input_ids.index(151653)        # Qwen vision end token
    prompt_len = len(input_ids)

    # 3. Model Generation
    # We MUST request output_hidden_states and output_attentions
    outputs = model.generate(
        **inputs, 
        max_new_tokens=64, 
        use_cache=True,
        output_hidden_states=True, 
        output_attentions=True,     
        return_dict_in_generate=True,
    )

    generated_ids = outputs.sequences[0][prompt_len:].tolist()
    token_labels = [processor.tokenizer.decode([tid]) for tid in generated_ids]
    n_steps = len(generated_ids)
    n_layers = len(outputs.hidden_states[0])

    print(f"  -> Generated {n_steps} tokens.")

    # =========================================================
    # A. ATTENTION TRIMODAL HEATMAP (Where is the token looking?)
    # =========================================================
    mat_v_attn = np.zeros((n_steps, n_layers))
    mat_p_attn = np.zeros((n_steps, n_layers))
    mat_h_attn = np.zeros((n_steps, n_layers))

    for step in range(n_steps):
        for l in range(n_layers):
            # With use_cache=True, attention at step > 0 is only relative to the new token
            attn = outputs.attentions[step][l][0] 
            
            # Average across all attention heads
            agg_attn = attn[:, -1, :].mean(dim=0).float().cpu().numpy()
            seq_len = len(agg_attn)
            
            # Sum of attention weights over the three main compartments
            v_val = agg_attn[img_start:img_end].sum()
            p_val = agg_attn[:img_start].sum() + agg_attn[img_end:prompt_len].sum()
            
            # Exclude self-loop (the last element of the sequence)
            h_val = agg_attn[prompt_len:seq_len-1].sum() if seq_len > prompt_len + 1 else 0.0
            
            total = v_val + p_val + h_val + 1e-12
            mat_v_attn[step, l] = v_val / total
            mat_p_attn[step, l] = p_val / total
            mat_h_attn[step, l] = h_val / total

    plot_trimodal_matrix(
        mat_v_attn, mat_p_attn, mat_h_attn, 
        os.path.join(out_dir, f"{stem}_attention_trimodal.png"), 
        token_labels, 
        title=f"Self-Attention Trimodal ({stem})"
    )

    # =========================================================
    # B. LOGIT LENS TRIMODAL HEATMAP (What is predicting the token?)
    # =========================================================
    mat_v_ll = np.zeros((n_steps, n_layers))
    mat_p_ll = np.zeros((n_steps, n_layers))
    mat_h_ll = np.zeros((n_steps, n_layers))

    context_hs = {l: [] for l in range(n_layers)}

    for step in range(n_steps):
        target_token_id = generated_ids[step]
        
        for l in range(n_layers):
            # Extract and concatenate the history of hidden states (bypass cache limitation)
            hs = outputs.hidden_states[step][l]
            context_hs[l].append(hs)
            full_hs = torch.cat(context_hs[l], dim=1) 
            
            # Apply final normalization if found
            if norm is not None:
                full_hs = norm(full_hs)
            
            # Project to vocabulary logit space
            with torch.no_grad():
                logits = lm_head(full_hs) 
            
            # How much probability do all past tokens assign to THIS generated token?
            scores = logits[0, :, target_token_id].float().cpu().numpy()
            
            # Keep only positive contributions (ReLU-like approach as in dodocore)
            scores_pos = np.maximum(scores, 0)
            seq_len = len(scores_pos)
            
            v_val = scores_pos[img_start:img_end].sum()
            p_val = scores_pos[:img_start].sum() + scores_pos[img_end:prompt_len].sum()
            h_val = scores_pos[prompt_len:seq_len-1].sum() if seq_len > prompt_len + 1 else 0.0
            
            total = v_val + p_val + h_val + 1e-12
            mat_v_ll[step, l] = v_val / total
            mat_p_ll[step, l] = p_val / total
            mat_h_ll[step, l] = h_val / total

    plot_trimodal_matrix(
        mat_v_ll, mat_p_ll, mat_h_ll, 
        os.path.join(out_dir, f"{stem}_logitlens_trimodal.png"), 
        token_labels, 
        title=f"Logit Lens Trimodal ({stem})"
    )


# ===========================================================================
# Main Execution Loop
# ===========================================================================

def main():
    input_dir = Path("test")                 # Folder containing your images
    out_dir = "imgs/trimodal_analysis"       # Output folder for the heatmaps
    prompt = "Describe this image in detail."
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Supported image extensions
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    images_to_process = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_exts]
    
    if not images_to_process:
        print(f"No valid images found in '{input_dir}' directory.")
        return

    print(f"Found {len(images_to_process)} images to process.")
    print("Loading Model into memory (this happens only once)...")
    
    model_name = "Qwen/Qwen2-VL-2B-Instruct"
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype="auto", device_map="cuda"
    )
    processor = AutoProcessor.from_pretrained(model_name)
    
    norm = _get_final_norm(model)
    if norm is None:
        print("[WARNING] Final norm not found, Logit Lens might be noisy.")
    else:
        print(f"[OK] Final norm successfully located: {type(norm).__name__}")
        
    print("-" * 50)
    
    # Process images in batch
    for i, img_path in enumerate(images_to_process, 1):
        print(f"[{i}/{len(images_to_process)}] Processing: {img_path.name}")
        analyze_single_image(model, processor, norm, img_path, prompt, out_dir)
        
    print("-" * 50)
    print(f"All done! Results are saved in: {out_dir}/")

if __name__ == "__main__":
    main()