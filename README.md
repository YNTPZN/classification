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

## Configuration

Edit `config.py` to modify:
- `DATA_ROOT` - Data root directory
- `MODEL_NAME` - Model (`efficientnet_b0` or `resnet50`)
- `IMG_SIZE` - Input image size
- `NUM_EPOCHS` - Number of training epochs
