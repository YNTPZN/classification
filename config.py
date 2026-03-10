"""Configuration for defect screening system."""

from pathlib import Path

# Data paths (relative to this file: cropping/ -> Classification/ -> Data/)
_SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = _SCRIPT_DIR.parent / "Data"
GOOD_FOLDER = "good"
DEFECT_PREFIX = "defect"  # All folders starting with "defect" are abnormal

# Output paths
OUTPUT_DIR = Path("/Users/xinyujia/Documents/Classification/cropping/output")
MODEL_SAVE_PATH = OUTPUT_DIR / "defect_classifier.pt"
RESULTS_DIR = OUTPUT_DIR / "screening_results"

# Model settings
MODEL_NAME = "efficientnet_b0"  # or "resnet50"
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4

# Training
VAL_SPLIT = 0.15
TEST_SPLIT = 0.1
RANDOM_SEED = 42

# Supported image extensions
IMG_EXTENSIONS = {".png", ".PNG", ".jpg", ".jpeg", ".JPG", ".JPEG"}
