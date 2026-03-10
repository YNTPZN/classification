"""
Defect Screening System

Separate good images from defect images, and mark suspicious defect regions in defect images.

Quick start:
  1. python train.py          # Train the model
  2. python screen.py         # Screen all images
  3. python screen.py --image path.png  # Screen a single image
"""

from pathlib import Path

# Re-export for convenience
from config import DATA_ROOT, GOOD_FOLDER, DEFECT_PREFIX, OUTPUT_DIR, MODEL_SAVE_PATH
