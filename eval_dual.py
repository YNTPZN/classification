#!/usr/bin/env python3
"""
Evaluate a saved dual-input finetuned model (dual_finetune.pt).

Usage example:
  python eval_dual.py \
    --tight-dir output/cropped_defects_tight_pad04 \
    --context-dir output/cropped_defects_context_pad04 \
    --ckpt output/dual_finetune.pt \
    --backbone convnext_tiny \
    --batch-size 64 \
    --device cpu
"""

import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import OUTPUT_DIR, RANDOM_SEED
from fewshot_dataset import load_cropped_defects, split_by_source_image_stratified, DualCroppedDefectDataset
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


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    per_cls = Counter()
    per_cls_ok = Counter()
    for (x1, x2), y, _ in loader:
        x1 = x1.to(device)
        x2 = x2.to(device)
        y = y.to(device)
        logits, _ = model(x1, x2)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        for yi, pi in zip(y.cpu().tolist(), pred.cpu().tolist()):
            per_cls[yi] += 1
            per_cls_ok[yi] += int(yi == pi)
    acc = correct / max(1, total)
    return acc, {c: per_cls_ok[c] / max(1, per_cls[c]) for c in per_cls}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tight-dir", type=Path, default=TIGHT_ROOT)
    p.add_argument("--context-dir", type=Path, default=CTX_ROOT)
    p.add_argument("--ckpt", type=Path, default=CKPT_DEFAULT)
    p.add_argument("--backbone", choices=["convnext_tiny", "resnet50"], default="convnext_tiny")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.15)
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

    samples, label_names = load_cropped_defects(args.tight_dir)
    n_classes = len(label_names)
    print(f"Classes: {label_names}")
    print(f"Total samples: {len(samples)}")

    _, _, test_s = split_by_source_image_stratified(samples, args.train_ratio, args.val_ratio, args.seed)
    print(f"Test samples: {len(test_s)}")

    test_ds = DualCroppedDefectDataset(test_s, context_root=args.context_dir)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    ckpt = torch.load(args.ckpt, map_location=device)
    backbone = ckpt.get("backbone", args.backbone)
    saved_labels = ckpt.get("label_names", label_names)

    if list(saved_labels) != list(label_names):
        print("Warning: label_names in checkpoint differ from current data:")
        print("  ckpt:", saved_labels)
        print("  data:", label_names)

    from dual_model import DualFusionNet as _DualFusionNet
    model = _DualFusionNet(backbone, num_classes=n_classes, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_acc, per = evaluate(model, test_loader, device)
    print("\n=== Dual model evaluation ===")
    print(f"Test accuracy: {test_acc:.4f}")
    print("Per-class accuracy (test):")
    for i, name in enumerate(label_names):
        if i in per:
            print(f"  {name}: {per[i]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

