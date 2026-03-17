#!/usr/bin/env python3
"""
Image-level evaluation for dual-input finetuned model.

For each original defect image, we:
  - use its tight/context crops from disk
  - run the dual model on all patches
  - aggregate patch predictions to one image-level label (mean logits)

Usage example:
  python eval_dual_image.py \
    --tight-dir output/cropped_defects_tight_pad04 \
    --context-dir output/cropped_defects_context_pad04 \
    --ckpt output/dual_finetune.pt \
    --backbone convnext_tiny \
    --batch-size 64 \
    --device cpu
"""

import argparse
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

from config import OUTPUT_DIR, RANDOM_SEED
from fewshot_dataset import load_cropped_defects, _get_source_stem
from dual_model import DualFusionNet


TIGHT_ROOT = OUTPUT_DIR / "cropped_defects_tight_pad04"
CTX_ROOT = OUTPUT_DIR / "cropped_defects_context_pad04"
CKPT_DEFAULT = OUTPUT_DIR / "dual_finetune.pt"


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_transform(img_size: int = 224):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def eval_image_level(
    model: DualFusionNet,
    tight_dir: Path,
    context_dir: Path,
    batch_size: int,
    device: torch.device,
):
    model.eval()

    samples, label_names = load_cropped_defects(tight_dir)
    n_classes = len(label_names)

    # Group crops by original image (label_idx, source_stem)
    groups = defaultdict(list)
    for p, label in samples:
        stem = _get_source_stem(p)
        groups[(label, stem)].append(p)

    transform = make_transform()

    correct = 0
    total = 0
    per_cls = Counter()
    per_cls_ok = Counter()

    group_items = list(groups.items())
    num_images = len(group_items)

    for idx, ((label, stem), paths) in enumerate(group_items):
        # Collect logits over all patches for this image
        logits_list = []
        # Iterate in mini-batches if many patches
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i : i + batch_size]
            tight_list = []
            ctx_list = []
            for p in batch_paths:
                cls_folder = p.parent.name
                ctx_path = context_dir / cls_folder / p.name
                if not ctx_path.exists():
                    continue
                img_tight = Image.open(p).convert("RGB")
                img_ctx = Image.open(ctx_path).convert("RGB")
                tight_list.append(transform(img_tight))
                ctx_list.append(transform(img_ctx))
            if not tight_list:
                continue
            x1 = torch.stack(tight_list, dim=0).to(device)
            x2 = torch.stack(ctx_list, dim=0).to(device)
            lg, _ = model(x1, x2)
            logits_list.append(lg.cpu())

        if not logits_list:
            # No valid patches with context; skip this image
            continue

        all_logits = torch.cat(logits_list, dim=0)  # (P, C)
        # Aggregate by mean logits, then argmax
        img_logits = all_logits.mean(dim=0, keepdim=True)
        pred = int(img_logits.argmax(dim=1).item())

        total += 1
        per_cls[label] += 1
        if pred == label:
            correct += 1
            per_cls_ok[label] += 1

        # Simple progress print every 50 images
        if (idx + 1) % 50 == 0 or (idx + 1) == num_images:
            print(f"Processed {idx+1}/{num_images} images...", flush=True)

    acc = correct / max(1, total)
    per_class_acc = {c: per_cls_ok[c] / max(1, per_cls[c]) for c in per_cls}
    return acc, per_class_acc, label_names


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tight-dir", type=Path, default=TIGHT_ROOT)
    p.add_argument("--context-dir", type=Path, default=CTX_ROOT)
    p.add_argument("--ckpt", type=Path, default=CKPT_DEFAULT)
    p.add_argument("--backbone", choices=["convnext_tiny", "resnet50"], default="convnext_tiny")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = p.parse_args()

    if not args.ckpt.exists():
        print(f"Checkpoint not found: {args.ckpt}")
        return 1

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")
    set_seed(args.seed)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    backbone = ckpt.get("backbone", args.backbone)
    saved_labels = ckpt.get("label_names", None)

    # Build model
    model = DualFusionNet(backbone, num_classes=len(saved_labels) if saved_labels else 1, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    acc, per, label_names = eval_image_level(
        model=model,
        tight_dir=args.tight_dir,
        context_dir=args.context_dir,
        batch_size=args.batch_size,
        device=device,
    )

    print("\n=== Image-level evaluation ===")
    print(f"Image-level accuracy: {acc:.4f}")
    print("Per-class (image-level) accuracy:")
    for i, name in enumerate(label_names):
        if i in per:
            print(f"  {name}: {per[i]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

