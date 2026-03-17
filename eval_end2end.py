#!/usr/bin/env python3
"""
End-to-end evaluation on a global test split (good + defect subclasses).

Pipeline per image:
  1) Good vs defect classifier (train.py -> defect_classifier.pt)
  2) If predicted good -> final label = good
  3) If predicted defect:
       - run Grad-CAM + boxes in memory
       - build tight/context crops
       - run dual finetuned model (dual_finetune.pt)
       - aggregate patch logits to one subclass label (mean logits)

Outputs:
  - overall accuracy (good + all defects)
  - good vs defect confusion
  - per-defect-subclass accuracy (on defect images only)
"""

import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

from config import DATA_ROOT, OUTPUT_DIR, MODEL_SAVE_PATH, MODEL_NAME, IMG_EXTENSIONS
from model import build_classifier, get_feature_layer
from gradcam import GradCAM, draw_defect_boxes
from dual_model import DualFusionNet


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_ROOT = SCRIPT_DIR.parent / "Dataset"
TRAIN_LIST = OUTPUT_DIR / "train_images.txt"
TEST_LIST = OUTPUT_DIR / "test_images.txt"


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_transform(img_size: int = 224):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def scale_boxes_to_original(boxes, cam_h, cam_w, orig_h, orig_w):
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


def expand_box(x1, y1, x2, y2, orig_w, orig_h, pad_ratio: float):
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


def load_binary_model(ckpt_path: Path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_name = ckpt.get("model_name", MODEL_NAME)
    model = build_classifier(model_name, num_classes=2)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, model_name


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    p.add_argument("--train-list", type=Path, default=TRAIN_LIST)
    p.add_argument("--test-list", type=Path, default=TEST_LIST)
    p.add_argument("--binary-ckpt", type=Path, default=MODEL_SAVE_PATH)
    p.add_argument("--dual-ckpt", type=Path, default=OUTPUT_DIR / "dual_finetune.pt")
    p.add_argument("--backbone", choices=["convnext_tiny", "resnet50"], default="convnext_tiny")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--context-pad", type=float, default=0.4)
    p.add_argument("--threshold", type=float, default=0.5, help="P(defect) threshold for binary classifier")
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    set_seed(args.seed)

    if not args.test_list.exists():
        print(f"Test list not found: {args.test_list}")
        print("Run split_dataset.py first.")
        return 1

    # Load binary classifier and Grad-CAM
    bin_model, bin_name = load_binary_model(args.binary_ckpt, device)
    target_layer = get_feature_layer(bin_model, bin_name)
    gradcam = GradCAM(bin_model, target_layer)
    transform = get_transform(args.img_size)

    # Load dual subclass model
    ckpt = torch.load(args.dual_ckpt, map_location=device, weights_only=False)
    label_names = ckpt.get("label_names", None)
    if not label_names:
        print("label_names missing in dual_finetune checkpoint.")
        return 1
    dual_model = DualFusionNet(args.backbone, num_classes=len(label_names), pretrained=False).to(device)
    dual_model.load_state_dict(ckpt["model_state_dict"])
    dual_model.eval()

    # Read test images list
    with args.test_list.open("r") as f:
        rel_paths = [line.strip() for line in f if line.strip()]

    # Good/defect confusion and subclass stats
    overall_correct = 0
    overall_total = 0

    good_defect_cm = Counter()  # (true, pred) pairs
    defect_per_cls = Counter()
    defect_per_cls_ok = Counter()

    for i, rel in enumerate(rel_paths):
        img_path = args.dataset_root / rel
        # Determine true label: good or defectX
        cls_name = img_path.parent.name  # e.g. good, defect3
        if cls_name == "good":
            true_bin = 0
            true_sub = None
        else:
            true_bin = 1
            # map defectX to subclass index based on label_names
            # label_names are defect1, defect2, ...
            if cls_name in label_names:
                true_sub = label_names.index(cls_name)
            else:
                true_sub = None

        img = Image.open(img_path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)

        # Step 1: good vs defect
        with torch.no_grad():
            logits = bin_model(x)
            probs = F.softmax(logits, dim=1)
            defect_prob = float(probs[0, 1].item())
            pred_bin = 1 if defect_prob >= args.threshold else 0

        # Record good/defect confusion
        good_defect_cm[(true_bin, pred_bin)] += 1

        if pred_bin == 0:
            # Predicted good
            pred_label = "good"
        else:
            # Predicted defect: run Grad-CAM to get boxes
            x_grad = transform(img).unsqueeze(0).to(device)
            x_grad.requires_grad_(True)
            cam, _ = gradcam.generate(x_grad, target_class=1)
            boxes = draw_defect_boxes(cam, threshold=0.4, min_area=50)
            if not boxes:
                # Fallback: treat as generic defect without subclass prediction
                pred_sub = None
            else:
                cam_h, cam_w = cam.shape[:2]
                img_np = np.array(img)
                orig_h, orig_w = img_np.shape[:2]
                scaled = scale_boxes_to_original(boxes, cam_h, cam_w, orig_h, orig_w)
                tight_list = []
                ctx_list = []
                for (x1, y1, x2, y2) in scaled:
                    crop = img_np[y1:y2, x1:x2]
                    cx1, cy1, cx2, cy2 = expand_box(x1, y1, x2, y2, orig_w, orig_h, args.context_pad)
                    ctx = img_np[cy1:cy2, cx1:cx2]
                    tight_list.append(transform(Image.fromarray(crop)))
                    ctx_list.append(transform(Image.fromarray(ctx)))
                if not tight_list:
                    pred_sub = None
                else:
                    x1 = torch.stack(tight_list, dim=0).to(device)
                    x2 = torch.stack(ctx_list, dim=0).to(device)
                    with torch.no_grad():
                        lg, _ = dual_model(x1, x2)  # (P, C)
                    img_logits = lg.mean(dim=0, keepdim=True)
                    pred_sub = int(img_logits.argmax(dim=1).item())

            if pred_sub is None:
                pred_label = "defect_unknown"
            else:
                pred_label = label_names[pred_sub]
                if true_sub is not None:
                    defect_per_cls[true_sub] += 1
                    if pred_sub == true_sub:
                        defect_per_cls_ok[true_sub] += 1

        # Overall correctness on full label space (good + subclasses)
        if cls_name == "good":
            overall_total += 1
            if pred_label == "good":
                overall_correct += 1
        else:
            overall_total += 1
            if pred_label == cls_name:
                overall_correct += 1

        if (i + 1) % 50 == 0 or (i + 1) == len(rel_paths):
            print(f"Processed {i+1}/{len(rel_paths)} images...", flush=True)

    overall_acc = overall_correct / max(1, overall_total)

    print("\n=== End-to-end evaluation on test split ===")
    print(f"Overall accuracy (good + subclasses): {overall_acc:.4f}")

    # Good vs defect confusion
    tn = good_defect_cm[(0, 0)]
    fp = good_defect_cm[(0, 1)]
    fn = good_defect_cm[(1, 0)]
    tp = good_defect_cm[(1, 1)]
    print("\nGood vs defect confusion (true, pred):")
    print(f"  TN (good->good):   {tn}")
    print(f"  FP (good->defect): {fp}")
    print(f"  FN (defect->good): {fn}")
    print(f"  TP (defect->defect): {tp}")

    # Subclass accuracy on defect images
    print("\nSubclass accuracy on defect images (image-level):")
    for idx, name in enumerate(label_names):
        if idx in defect_per_cls:
            acc = defect_per_cls_ok[idx] / max(1, defect_per_cls[idx])
            print(f"  {name}: {acc:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

