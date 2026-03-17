#!/usr/bin/env python3
"""
Plot prediction score distribution: good vs defect.
Shows defect probability (class 1) for each group to check separation.
"""

import matplotlib
matplotlib.use("Agg")  # Non-GUI backend for headless environments

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import DATA_ROOT, GOOD_FOLDER, DEFECT_PREFIX, OUTPUT_DIR, MODEL_SAVE_PATH, MODEL_NAME, IMG_SIZE, DEFECT_THRESHOLD
from dataset import load_dataset, DefectDataset
from model import build_classifier


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


def collect_scores(model, samples, transform, device, batch_size=32):
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
            defect_prob = probs[:, 1].cpu().numpy()  # P(defect)
            all_probs.extend(defect_prob.tolist())
            all_labels.extend(labels.tolist())

    return np.array(all_probs), np.array(all_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=MODEL_SAVE_PATH)
    parser.add_argument("--data", type=Path, default=DATA_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "score_distribution.png")
    args = parser.parse_args()

    if not args.model.exists():
        print(f"Model not found: {args.model}")
        return 1

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    model = load_model(args.model, device)
    transform = get_transform()

    samples, _ = load_dataset(args.data, GOOD_FOLDER, DEFECT_PREFIX)
    probs, labels = collect_scores(model, samples, transform, device, batch_size=64)

    good_probs = probs[labels == 0]
    defect_probs = probs[labels == 1]

    print(f"\n=== Score Statistics ===")
    print(f"Good   (n={len(good_probs)}): mean={good_probs.mean():.4f}, std={good_probs.std():.4f}, min={good_probs.min():.4f}, max={good_probs.max():.4f}")
    print(f"Defect (n={len(defect_probs)}): mean={defect_probs.mean():.4f}, std={defect_probs.std():.4f}, min={defect_probs.min():.4f}, max={defect_probs.max():.4f}")

    # Overlap at DEFECT_THRESHOLD
    n_good_high = (good_probs >= DEFECT_THRESHOLD).sum()
    n_defect_low = (defect_probs < DEFECT_THRESHOLD).sum()
    print(f"\nOverlap at threshold {DEFECT_THRESHOLD}:")
    print(f"  Good misclassified (prob >= {DEFECT_THRESHOLD}): {n_good_high} ({100*n_good_high/len(good_probs):.1f}%)")
    print(f"  Defect misclassified (prob < {DEFECT_THRESHOLD}): {n_defect_low} ({100*n_defect_low/len(defect_probs):.1f}%)")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: histograms
    ax1 = axes[0]
    bins = np.linspace(0, 1, 26)
    ax1.hist(good_probs, bins=bins, alpha=0.6, label="good", color="green", density=True, edgecolor="black", linewidth=0.5)
    ax1.hist(defect_probs, bins=bins, alpha=0.6, label="defect", color="red", density=True, edgecolor="black", linewidth=0.5)
    ax1.axvline(DEFECT_THRESHOLD, color="black", linestyle="--", linewidth=1, label=f"threshold {DEFECT_THRESHOLD}")
    ax1.set_xlabel("Defect probability (P(defect))")
    ax1.set_ylabel("Density")
    ax1.set_title("Score Distribution: Good vs Defect")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: box plot
    ax2 = axes[1]
    bp = ax2.boxplot(
        [good_probs, defect_probs],
        labels=["good", "defect"],
        patch_artist=True,
        showmeans=True,
    )
    bp["boxes"][0].set_facecolor("lightgreen")
    bp["boxes"][1].set_facecolor("lightcoral")
    ax2.axhline(DEFECT_THRESHOLD, color="black", linestyle="--", linewidth=1)
    ax2.set_ylabel("Defect probability")
    ax2.set_title("Score Box Plot")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nPlot saved to: {args.output}")
    return 0


if __name__ == "__main__":
    exit(main())
