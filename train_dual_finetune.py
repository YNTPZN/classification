#!/usr/bin/env python3
"""
Train a dual-input defect subclass classifier on cropped patches:
  - input A: tight crop  (output/cropped_defects/*)
  - input B: context crop (output/cropped_defects_context/*)

Backbone: ConvNeXt-Tiny or ResNet50
Loss: CrossEntropy + alpha * Supervised Contrastive
Sampler: class-balanced (WeightedRandomSampler)

Recommended first run (if you have CUDA):
  python train_dual_finetune.py --backbone convnext_tiny --epochs 10 --batch-size 64
"""

import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from config import OUTPUT_DIR, RANDOM_SEED
from fewshot_dataset import load_cropped_defects, split_by_source_image_stratified, DualCroppedDefectDataset
from dual_model import DualFusionNet
from losses import supervised_contrastive_loss


TIGHT_ROOT = OUTPUT_DIR / "cropped_defects"
CTX_ROOT = OUTPUT_DIR / "cropped_defects_context"


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_balanced_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    counts = Counter(labels.tolist())
    weights = np.array([1.0 / counts[int(y)] for y in labels], dtype=np.float32)
    return WeightedRandomSampler(weights=torch.from_numpy(weights), num_samples=len(weights), replacement=True)


@torch.no_grad()
def evaluate(model, loader, device, n_classes: int):
    model.eval()
    all_true = []
    all_pred = []
    correct = 0
    total = 0

    for (x1, x2), y, _ in loader:
        x1 = x1.to(device)
        x2 = x2.to(device)
        y = y.to(device)
        logits, _ = model(x1, x2)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        all_true.append(y.detach().cpu().numpy())
        all_pred.append(pred.detach().cpu().numpy())

    acc = correct / max(1, total)

    all_true = np.concatenate(all_true) if all_true else np.array([], dtype=np.int64)
    all_pred = np.concatenate(all_pred) if all_pred else np.array([], dtype=np.int64)
    class_metrics = {}
    for c in range(n_classes):
        tp = int(((all_pred == c) & (all_true == c)).sum())
        fp = int(((all_pred == c) & (all_true != c)).sum())
        fn = int(((all_pred != c) & (all_true == c)).sum())
        support = int((all_true == c).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        class_metrics[c] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    return acc, class_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tight-dir", type=Path, default=TIGHT_ROOT)
    p.add_argument("--context-dir", type=Path, default=CTX_ROOT)
    p.add_argument("--backbone", choices=["convnext_tiny", "resnet50"], default="convnext_tiny")
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--alpha", type=float, default=0.2, help="weight for SupCon loss")
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--save", type=Path, default=OUTPUT_DIR / "dual_finetune.pt")
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    args = p.parse_args()

    if not args.tight_dir.exists():
        print(f"Missing tight crops: {args.tight_dir}")
        return 1
    if not args.context_dir.exists():
        print(f"Missing context crops: {args.context_dir}")
        print("Generate them first: python defect_crop.py --save-context")
        return 1

    set_seed(args.seed)
    if args.device == "auto":
        # MPS can be unstable for some torchvision backbones/backward paths; prefer CPU unless CUDA exists.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    samples, label_names = load_cropped_defects(args.tight_dir)
    n_classes = len(label_names)
    print(f"Classes: {label_names}")
    print(f"Total samples: {len(samples)}")

    train_s, val_s, test_s = split_by_source_image_stratified(samples, args.train_ratio, args.val_ratio, args.seed)
    print(f"Split: train={len(train_s)}, val={len(val_s)}, test={len(test_s)}")

    train_labels = np.array([l for _, l in train_s], dtype=np.int64)
    sampler = make_balanced_sampler(train_labels)

    train_ds = DualCroppedDefectDataset(train_s, context_root=args.context_dir)
    val_ds = DualCroppedDefectDataset(val_s, context_root=args.context_dir)
    test_ds = DualCroppedDefectDataset(test_s, context_root=args.context_dir)

    num_workers = 4 if torch.cuda.is_available() else 0
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

    model = DualFusionNet(args.backbone, num_classes=n_classes, pretrained=args.pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ce = nn.CrossEntropyLoss()

    best_val = 0.0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for (x1, x2), y, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)
            opt.zero_grad()
            logits, z = model(x1, x2)
            loss_ce = ce(logits, y)
            loss_sc = supervised_contrastive_loss(z, y, temperature=args.temperature)
            loss = loss_ce + args.alpha * loss_sc
            loss.backward()
            opt.step()
            total_loss += loss.item()

        val_acc, _ = evaluate(model, val_loader, device, n_classes)
        print(f"Epoch {epoch+1}: loss={total_loss/max(1,len(train_loader)):.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            ckpt = {
                "model_state_dict": model.state_dict(),
                "label_names": label_names,
                "backbone": args.backbone,
            }
            torch.save(ckpt, args.save)
            print(f"  -> saved best to {args.save}")

    print("\nBest val acc:", best_val)
    test_acc, per_metrics = evaluate(model, test_loader, device, n_classes)
    print("Test acc:", test_acc)
    print("\nPer-class metrics (test):")
    for i, name in enumerate(label_names):
        if i in per_metrics:
            m = per_metrics[i]
            print(
                f"  {name}: precision={m['precision']:.4f}, recall={m['recall']:.4f}, "
                f"f1={m['f1']:.4f}, support={m['support']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

