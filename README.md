# Defect Screening System

A defect screening system that separates normal images from abnormal images and highlights suspicious defect regions in abnormal images.

## Data Format

- `good/` - Normal images
- `defect1/`, `defect2/`, ... - Abnormal images (all folders starting with `defect`)

Data path: `Data/` (relative to project root)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Train the Classifier

```bash
python train.py
```

Optional arguments:
- `--data` - Data root directory
- `--epochs` - Number of training epochs (default: 15)
- `--batch-size` - Batch size (default: 32)
- `--lr` - Learning rate (default: 1e-4)
- `--output` - Model save path

### 2. Screen Images

**Screen a single image:**
```bash
python screen.py --image /path/to/image.png
```

**Screen an entire directory:**
```bash
python screen.py --input-dir /path/to/images
```

**Screen all images in Data folder (good + defect folders):**
```bash
python screen.py
```

### 3. Output

- Classification result: `good` (normal) or `defect` (abnormal)
- Confidence: 0–100%
- For abnormal images, the system generates:
  - `*_defect_overlay.png` - Heatmap overlaid on original image
  - `*_defect_heatmap.png` - Defect region heatmap
- Bounding box coordinates for suspicious regions

### 4. Few-Shot Defect Classification (on cropped patches)

After cropping defect regions (`python defect_crop.py`), classify crop images into defect subclasses using DINOv2 features + ProtoNet / cosine / kNN:

```bash
# ProtoNet (nearest prototype, default)
python fewshot_classifier.py

# Cosine classifier
python fewshot_classifier.py --method cosine

# k-NN
python fewshot_classifier.py --method knn --k 5
```

- **Data**: `output/cropped_defects/defect1/`, `defect2/`, ... (folder name = label)
- **Backbone**: DINOv2 ViT-B/14 (pretrained, no fine-tuning)
- **Methods**: ProtoNet (class prototype mean), cosine similarity, k-NN
- **Save prototypes**: `python fewshot_classifier.py --save-prototypes`

## Configuration

Edit `config.py` to modify:
- `DATA_ROOT` - Data root directory
- `MODEL_NAME` - Model (`efficientnet_b0` or `resnet50`)
- `IMG_SIZE` - Input image size
- `NUM_EPOCHS` - Number of training epochs
