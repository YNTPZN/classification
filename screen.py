#!/usr/bin/env python3
"""
Defect screening system: classify good vs defect, localize defect regions.
Usage:
  python screen.py                    # Screen all images in Data folder
  python screen.py --image path.png   # Screen single image
  python screen.py --input-dir /path  # Screen images in directory
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

from config import (
    DATA_ROOT,
    GOOD_FOLDER,
    DEFECT_PREFIX,
    OUTPUT_DIR,
    MODEL_SAVE_PATH,
    RESULTS_DIR,
    MODEL_NAME,
    IMG_SIZE,
    IMG_EXTENSIONS,
)
from dataset import load_dataset, get_image_paths, DefectDataset
from model import build_classifier, get_feature_layer
from gradcam import GradCAM, overlay_heatmap, get_defect_regions, draw_defect_boxes


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


def screen_image(model, gradcam, img_path: Path, transform, device, save_dir: Path = None):
    """
    Screen single image. Returns (prediction, confidence, cam, overlay).
    prediction: 0=good, 1=defect
    """
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1)
        pred = logits.argmax(dim=1).item()
        conf = probs[0, pred].item()
    
    cam = None
    overlay = None
    boxes = []
    
    if pred == 1:  # Defect - generate Grad-CAM
        img_tensor_grad = transform(img).unsqueeze(0).to(device)
        img_tensor_grad.requires_grad_(True)
        cam, _ = gradcam.generate(img_tensor_grad, target_class=1)
        overlay, _ = overlay_heatmap(cam, img, alpha=0.5)
        boxes = draw_defect_boxes(cam, threshold=0.4, min_area=50)
    
    result = {
        "path": str(img_path),
        "prediction": pred,
        "label_name": "good" if pred == 0 else "defect",
        "confidence": conf,
        "cam": cam,
        "overlay": overlay,
        "boxes": boxes,
        "original_img": np.array(img),
    }
    
    if save_dir and pred == 1 and overlay is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        name = img_path.stem
        # Save overlay
        overlay_pil = Image.fromarray(overlay)
        overlay_pil.save(save_dir / f"{name}_defect_overlay.png")
        # Save CAM heatmap
        plt.figure(figsize=(6, 6))
        plt.imshow(cam, cmap="jet")
        plt.colorbar()
        plt.axis("off")
        plt.title(f"Defect regions - {img_path.name}")
        plt.savefig(save_dir / f"{name}_defect_heatmap.png", bbox_inches="tight")
        plt.close()
    
    return result


def screen_folder(model, gradcam, folder: Path, transform, device, save_dir: Path):
    """Screen all images in folder."""
    paths = []
    for ext in IMG_EXTENSIONS:
        paths.extend(folder.glob(f"*{ext}"))
    
    results = []
    for p in sorted(paths):
        try:
            r = screen_image(model, gradcam, p, transform, device, save_dir)
            results.append(r)
        except Exception as e:
            print(f"Error processing {p}: {e}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Defect screening system")
    parser.add_argument("--model", type=Path, default=MODEL_SAVE_PATH)
    parser.add_argument("--image", type=Path, help="Single image to screen")
    parser.add_argument("--input-dir", type=Path, help="Directory of images to screen")
    parser.add_argument("--data", type=Path, default=DATA_ROOT, help="Data root (for full screening)")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR)
    parser.add_argument("--no-save", action="store_true", help="Don't save defect visualizations")
    args = parser.parse_args()
    
    if not args.model.exists():
        print(f"Model not found: {args.model}")
        print("Please run: python train.py")
        return 1
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model, model_name = load_model(args.model, device)
    target_layer = get_feature_layer(model, model_name)
    gradcam = GradCAM(model, target_layer)
    transform = get_transform()
    
    save_dir = None if args.no_save else args.output
    
    if args.image:
        # Single image
        r = screen_image(model, gradcam, args.image, transform, device, save_dir)
        print(f"\nImage: {args.image.name}")
        print(f"  Prediction: {r['label_name']} (confidence: {r['confidence']:.2%})")
        if r["prediction"] == 1:
            print(f"  Defect regions: {len(r['boxes'])} box(es) detected")
            if r["boxes"]:
                for i, (x1, y1, x2, y2) in enumerate(r["boxes"]):
                    print(f"    Box {i+1}: ({x1}, {y1}) - ({x2}, {y2})")
        return 0
    
    if args.input_dir:
        # Screen directory
        results = screen_folder(model, gradcam, args.input_dir, transform, device, save_dir or args.output)
    else:
        # Screen all: good + defect folders
        results = []
        for folder in [args.data / GOOD_FOLDER] + sorted([d for d in args.data.iterdir() if d.is_dir() and d.name.startswith(DEFECT_PREFIX)]):
            folder_results = screen_folder(model, gradcam, folder, transform, device, save_dir)
            results.extend(folder_results)
    
    # Summary
    n_total = len(results)
    n_good = sum(1 for r in results if r["prediction"] == 0)
    n_defect = sum(1 for r in results if r["prediction"] == 1)
    
    print(f"\n=== Screening Summary ===")
    print(f"Total: {n_total} | Predicted Good: {n_good} | Predicted Defect: {n_defect}")
    
    if save_dir:
        print(f"Defect visualizations saved to: {save_dir}")
    
    return 0


if __name__ == "__main__":
    exit(main())
