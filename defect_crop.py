#!/usr/bin/env python3
"""
Crop defect regions from images in defect folders using trained classifier + Grad-CAM.

Usage:
  python defect_crop.py                    # Crop all defect folders under Data/
  python defect_crop.py --input-dir /path   # Crop specific directory
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from config import (
    DATA_ROOT,
    DEFECT_PREFIX,
    OUTPUT_DIR,
    MODEL_SAVE_PATH,
    MODEL_NAME,
    IMG_SIZE,
    IMG_EXTENSIONS,
)
from model import build_classifier, get_feature_layer
from gradcam import GradCAM, draw_defect_boxes


CROPPED_DIR = OUTPUT_DIR / "cropped_defects"
CONTEXT_DIR = OUTPUT_DIR / "cropped_defects_context"
TRAIN_LIST = OUTPUT_DIR / "train_images.txt"


def get_transform():
    import torchvision.transforms as T
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def load_model(ckpt_path: Path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_name = ckpt.get("model_name", MODEL_NAME)
    model = build_classifier(model_name, num_classes=2)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, model_name


def scale_boxes_to_original(boxes, cam_h, cam_w, orig_h, orig_w):
    """Scale boxes from CAM size (224x224) to original image size."""
    scale_x = orig_w / cam_w
    scale_y = orig_h / cam_h
    scaled = []
    for x1, y1, x2, y2 in boxes:
        x1_s = max(0, int(x1 * scale_x))
        y1_s = max(0, int(y1 * scale_y))
        x2_s = min(orig_w, int(x2 * scale_x))
        y2_s = min(orig_h, int(y2 * scale_y))
        if x2_s > x1_s and y2_s > y1_s:
            scaled.append((x1_s, y1_s, x2_s, y2_s))
    return scaled


def _expand_box(x1, y1, x2, y2, orig_w, orig_h, pad_ratio: float):
    """Expand box by pad_ratio of its size (clamped)."""
    w = x2 - x1
    h = y2 - y1
    pad_x = int(round(w * pad_ratio))
    pad_y = int(round(h * pad_ratio))
    ex1 = max(0, x1 - pad_x)
    ey1 = max(0, y1 - pad_y)
    ex2 = min(orig_w, x2 + pad_x)
    ey2 = min(orig_h, y2 + pad_y)
    if ex2 <= ex1 or ey2 <= ey1:
        return x1, y1, x2, y2
    return ex1, ey1, ex2, ey2


def crop_defect_regions(
    img_path: Path,
    model,
    gradcam,
    transform,
    device,
    save_dir: Path,
    context_dir: Path | None = None,
    context_pad: float = 0.5,
):
    """
    Process single image: classify, get defect boxes, crop and save.
    Returns number of crops saved.
    """
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)
    orig_h, orig_w = img_np.shape[:2]

    # Defect folders contain only defect images - always run Grad-CAM (no classification filter)
    img_tensor_grad = transform(img).unsqueeze(0).to(device)
    img_tensor_grad.requires_grad_(True)
    cam, _ = gradcam.generate(img_tensor_grad, target_class=1)
    boxes = draw_defect_boxes(cam, threshold=0.4, min_area=50)

    if not boxes:
        return 0

    # Scale boxes to original image size
    cam_h, cam_w = cam.shape[:2]
    scaled_boxes = scale_boxes_to_original(boxes, cam_h, cam_w, orig_h, orig_w)

    save_dir.mkdir(parents=True, exist_ok=True)
    stem = img_path.stem

    for i, (x1, y1, x2, y2) in enumerate(scaled_boxes):
        out_name = f"{stem}_crop_{i}.png"
        # Tight crop
        crop = img_np[y1:y2, x1:x2]
        Image.fromarray(crop).save(save_dir / out_name)

        # Context crop (expanded box)
        if context_dir is not None:
            cx1, cy1, cx2, cy2 = _expand_box(x1, y1, x2, y2, orig_w, orig_h, pad_ratio=context_pad)
            ctx = img_np[cy1:cy2, cx1:cx2]
            (context_dir).mkdir(parents=True, exist_ok=True)
            Image.fromarray(ctx).save(context_dir / out_name)

    return len(scaled_boxes)


def main():
    parser = argparse.ArgumentParser(description="Crop defect regions from images")
    parser.add_argument("--model", type=Path, default=MODEL_SAVE_PATH)
    parser.add_argument("--data", type=Path, default=DATA_ROOT)
    parser.add_argument("--input-dir", type=Path, help="Single defect folder to process")
    parser.add_argument("--output", type=Path, default=CROPPED_DIR)
    parser.add_argument("--save-context", action="store_true", help="Also save expanded context crops")
    parser.add_argument("--context-output", type=Path, default=CONTEXT_DIR)
    parser.add_argument("--context-pad", type=float, default=0.5, help="Padding ratio for context crop expansion")
    args = parser.parse_args()

    if not args.model.exists():
        print(f"Model not found: {args.model}")
        print("Please run: python train.py")
        return 1

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    model, model_name = load_model(args.model, device)
    target_layer = get_feature_layer(model, model_name)
    gradcam = GradCAM(model, target_layer)
    transform = get_transform()

    args.output.mkdir(parents=True, exist_ok=True)

    # Determine which defect images to process
    train_defect_paths = None
    if TRAIN_LIST.exists():
        with TRAIN_LIST.open("r") as f:
            rel_train = [Path(line.strip()) for line in f if line.strip()]
        # Keep only defect folders from the global train list
        train_defect_paths = {
            args.data / rel for rel in rel_train if rel.parts and rel.parts[0].startswith(DEFECT_PREFIX)
        }

    if args.input_dir:
        folders = [args.input_dir]
    else:
        folders = sorted(
            d for d in args.data.iterdir()
            if d.is_dir() and d.name.startswith(DEFECT_PREFIX)
        )

    if not folders:
        print(f"No defect folders found under {args.data}")
        return 1

    total_images = 0
    total_crops = 0

    for folder in folders:
        paths = []
        for ext in IMG_EXTENSIONS:
            for p in folder.glob(f"*{ext}"):
                if train_defect_paths is not None and p not in train_defect_paths:
                    continue
                paths.append(p)
        paths = sorted(paths)

        if not paths:
            continue

        save_dir = args.output / folder.name
        ctx_dir = (args.context_output / folder.name) if args.save_context else None
        n_crops = 0

        for p in tqdm(paths, desc=folder.name, leave=False):
            try:
                n = crop_defect_regions(
                    p,
                    model,
                    gradcam,
                    transform,
                    device,
                    save_dir,
                    context_dir=ctx_dir,
                    context_pad=args.context_pad,
                )
                if n > 0:
                    total_images += 1
                    n_crops += n
            except Exception as e:
                print(f"Error processing {p}: {e}")

        total_crops += n_crops
        if n_crops > 0:
            print(f"  {folder.name}: {len(paths)} images -> {n_crops} crops")

    print(f"\n=== Cropping Complete ===")
    print(f"Images with defects: {total_images}")
    print(f"Total crops saved: {total_crops}")
    print(f"Output: {args.output}")
    if args.save_context:
        print(f"Context output: {args.context_output} (pad={args.context_pad})")

    return 0


if __name__ == "__main__":
    exit(main())
