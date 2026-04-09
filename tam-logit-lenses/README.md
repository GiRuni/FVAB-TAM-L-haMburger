# Logit Lens and Spatial Reasoning Support for TAM for MultiModal Large Language Models
 
This repo extends the original [Token Activation Map (TAM)](https://github.com/xmed-lab/TAM) with logit-lens support and a spatial evaluation pipeline. For the base TAM method, demos, and datasets refer to the original repo.
 
---
 
## File needed for testing
 
| File | Purpose |
|---|---|
| `demo.py` | Original TAM demo with logit-lens across all layers and generate per-token heatmap grids - three models are employed in the demo for now.|
| `new_eval.py` | Evaluate TAM heatmaps against binary object masks and spatial relation masks - not only based on TAM original metrics but also on three other, check below for more information. |
 
---
 
## Setup
 
```bash
pip install -r requirements.txt
```
Just run the two files separately:
 
| `demo.py` | For producing heatmaps for every token from every layer through TAM.
```bash
python demo.py
```
---
 
| `new_eval.py` | Measuting metrics for heatmaps produced, change in the configuration section according to tests.
```bash
python new_eval.py
```
---
 
 
Metrics computed per token step and layer:
 
| Metric | Description |
|---|---|
| `obj_iou` | Heatmaps IoU against object mask (Original TAM)|
| `func_iou` | Functional words heatmap agains background (Original TAM)|
| `f1_iou` | Harmonic mean of obj_iou and func_iou (Original TAM) |
| `iou_hard` | Hard IoU at 0.5 threshold |
| `io_ratio` | Fraction of heatmap mass inside the GT mask [Q-GroundCAM](https://arxiv.org/abs/2404.19128) |
| `wdp` | Distance-weighted penalty for activation outside GT mask [Q-GroundCAM](https://arxiv.org/abs/2404.19128) |
 
Outputs saved to `results/`:
- `results.csv` — one row per (image, layer, step, target)
- `summary.csv` — averaged per (target_type, target, layer)
- `img/ll_` — directory with all heatmaps produced, The directory endin with `_token_grids` collects a grid of all heatmaps from layer for each token generated.
 
Spatial relations are detected automatically from the generated text using `rel_config.yaml` and evaluated against all available mask pairs for that image.
