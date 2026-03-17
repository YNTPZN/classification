#!/usr/bin/env python3
"""
Evaluate model metrics: compare defect_classifier.pt (threshold 0.4) vs defect_classifier_5.pt (threshold 0.5).
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import DATA_ROOT, GOOD_FOLDER, DEFECT_PREFIX, OUTPUT_DIR, MODEL_SAVE_PATH, MODEL_NAME, IMG_SIZE
from dataset import load_dataset, DefectDataset
from model import build_classifier

MODEL_04 = OUTPUT_DIR / "defect_classifier.pt"      # threshold 0.4
MODEL_05 = OUTPUT_DIR / "defect_classifier_5.pt"    # threshold 0.5


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
    return model


def collect_scores(model, samples, transform, device, batch_size=64):
    """Run inference and collect defect (class 1) probabilities."""
    dataset = DefectDataset(samples, transform=transform, img_size=IMG_SIZE, is_training=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels, _ in tqdm(loader, desc="Inference"):
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = F.softmax(logits, dim=1)
            defect_prob = probs[:, 1].cpu().numpy()
            all_probs.extend(defect_prob.tolist())
            all_labels.extend(labels.tolist())

    return np.array(all_probs), np.array(all_labels)


def compute_metrics(probs, labels, threshold):
    """Compute TP, TN, FP, FN and derived metrics at given threshold."""
    preds = (probs >= threshold).astype(int)
    # label: 0=good, 1=defect
    TP = ((preds == 1) & (labels == 1)).sum()
    TN = ((preds == 0) & (labels == 0)).sum()
    FP = ((preds == 1) & (labels == 0)).sum()
    FN = ((preds == 0) & (labels == 1)).sum()

    n = len(labels)
    accuracy = (TP + TN) / n if n > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "TP": int(TP), "TN": int(TN), "FP": int(FP), "FN": int(FN),
        "Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1": f1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-04", type=Path, default=MODEL_04, help="Model for threshold 0.4")
    parser.add_argument("--model-05", type=Path, default=MODEL_05, help="Model for threshold 0.5")
    parser.add_argument("--data", type=Path, default=DATA_ROOT)
    args = parser.parse_args()

    if not args.model_04.exists():
        print(f"Model not found: {args.model_04}")
        return 1
    if not args.model_05.exists():
        print(f"Model not found: {args.model_05}")
        return 1

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    transform = get_transform()
    samples, _ = load_dataset(args.data, GOOD_FOLDER, DEFECT_PREFIX)

    n_total = len(samples)
    n_good = sum(1 for _, l in samples if l == 0)
    n_defect = sum(1 for _, l in samples if l == 1)
    print(f"\nDataset: {n_total} total (good={n_good}, defect={n_defect})")

    # Model 0.4 (defect_classifier.pt) @ threshold 0.4
    print("\nEvaluating defect_classifier.pt (threshold 0.4)...")
    model04 = load_model(args.model_04, device)
    probs04, labels = collect_scores(model04, samples, transform, device)
    m04 = compute_metrics(probs04, labels, 0.4)

    # Model 0.5 (defect_classifier_5.pt) @ threshold 0.5
    print("\nEvaluating defect_classifier_5.pt (threshold 0.5)...")
    model05 = load_model(args.model_05, device)
    probs05, _ = collect_scores(model05, samples, transform, device)
    m05 = compute_metrics(probs05, labels, 0.5)

    print("\n" + "=" * 65)
    print("defect_classifier.pt (0.4) vs defect_classifier_5.pt (0.5)")
    print("=" * 65)

    print("\n--- Confusion Matrix ---")
    print(f"{'':>12} {'defect_classifier.pt':>22} {'defect_classifier_5.pt':>22}")
    print(f"{'':>12} {'(threshold 0.4)':>22} {'(threshold 0.5)':>22}")
    print(f"{'TP':>10} {m04['TP']:>22} {m05['TP']:>22}")
    print(f"{'TN':>10} {m04['TN']:>22} {m05['TN']:>22}")
    print(f"{'FP':>10} {m04['FP']:>22} {m05['FP']:>22}")
    print(f"{'FN':>10} {m04['FN']:>22} {m05['FN']:>22}")

    print("\n--- Metrics ---")
    print(f"{'':>12} {'defect_classifier.pt':>22} {'defect_classifier_5.pt':>22}")
    print(f"{'Accuracy':>10} {m04['Accuracy']:>21.4f} {m05['Accuracy']:>21.4f}")
    print(f"{'Precision':>10} {m04['Precision']:>21.4f} {m05['Precision']:>21.4f}")
    print(f"{'Recall':>10} {m04['Recall']:>21.4f} {m05['Recall']:>21.4f}")
    print(f"{'F1':>10} {m04['F1']:>21.4f} {m05['F1']:>21.4f}")

    print("\n" + "=" * 65)
    return 0


if __name__ == "__main__":
    exit(main())
