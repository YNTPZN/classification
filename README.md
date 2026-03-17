# Defect Screening System

## Installation

```bash
pip install -r requirements.txt
```

## Commands

### Train

```bash
python train.py
```

### Screen

```bash
python screen.py --image /path/to/image.png
```

### Crop

```bash
python defect_crop.py \
  --save-context \
  --context-pad 0.4 \
  --output output/cropped_defects_tight_pad04 \
  --context-output output/cropped_defects_context_pad04
```

### Diagnostics

```bash
python plot_score_dist.py
```

```bash
python eval_metrics.py
```

### Few-shot (DINOv2 features)

```bash
python fewshot_classifier.py --method knn --k 1 --knn-metric cosine --split-by-image --stratified
```
### Dual-input fine-tuning (ConvNeXt/ResNet)

```bash
python train_dual_finetune.py \
  --tight-dir output/cropped_defects_tight_pad04 \
  --context-dir output/cropped_defects_context_pad04 \
  --backbone convnext_tiny \
  --epochs 10 \
  --batch-size 64 \
  --device gpu
```
