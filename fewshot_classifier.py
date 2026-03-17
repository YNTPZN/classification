#!/usr/bin/env python3
"""
Few-shot defect classification: DINOv2 features + ProtoNet / cosine / kNN.

Usage:
  python fewshot_classifier.py                    # Train & eval with default settings
  python fewshot_classifier.py --method knn       # Use kNN instead of protonet
  python fewshot_classifier.py --method cosine    # Use cosine classifier
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import OUTPUT_DIR
from fewshot_dataset import (
    load_cropped_defects,
    CroppedDefectDataset,
    DualCroppedDefectDataset,
    split_by_source_image,
    split_by_source_image_stratified,
    DINOv2_IMG_SIZE,
)


CROPPED_DIR = OUTPUT_DIR / "cropped_defects"
CONTEXT_DIR = OUTPUT_DIR / "cropped_defects_context"
PROTOTYPE_DIR = OUTPUT_DIR / "fewshot_prototypes"
RANDOM_SEED = 42


def get_dinov2_backbone(model_size: str = "vitb14", device=None):
    """Load DINOv2 backbone for feature extraction."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # MPS has bicubic issues with DINOv2
    model = torch.hub.load("facebookresearch/dinov2", f"dinov2_{model_size}")
    model.to(device)
    model.eval()
    return model, device


def _random_flip_batch(imgs: torch.Tensor) -> torch.Tensor:
    """Cheap TTA: random H/V flips on a batch tensor (B,C,H,W)."""
    out = imgs
    if torch.rand(()) < 0.5:
        out = torch.flip(out, dims=[3])  # horizontal
    if torch.rand(()) < 0.5:
        out = torch.flip(out, dims=[2])  # vertical
    return out


def extract_features(model, loader, device):
    """Extract DINOv2 features (CLS token) for all samples. Supports dual input (tight, context)."""
    model.eval()
    all_features = []
    all_labels = []
    all_paths = []

    with torch.no_grad():
        for imgs, labels, paths in tqdm(loader, desc="Extract features"):
            # imgs can be Tensor (B,C,H,W) or tuple(tight, ctx)
            if isinstance(imgs, (tuple, list)) and len(imgs) == 2:
                tight, ctx = imgs
                tight = tight.to(device)
                ctx = ctx.to(device)
                out1 = model(tight)
                out2 = model(ctx)

                def _to_cls(out):
                    if isinstance(out, dict):
                        f = out.get("x", out.get("cls_token", list(out.values())[0]))
                    else:
                        f = out
                    if f.dim() == 3:
                        f = f[:, 0]
                    return f

                f1 = _to_cls(out1)
                f2 = _to_cls(out2)
                # Fuse by concatenation (tight || context)
                feats = torch.cat([f1, f2], dim=1)
            else:
                imgs = imgs.to(device)
                out = model(imgs)
                if isinstance(out, dict):
                    feats = out.get("x", out.get("cls_token", list(out.values())[0]))
                else:
                    feats = out
                if feats.dim() == 3:  # (B, N, D) -> CLS is first token
                    feats = feats[:, 0]

            all_features.append(feats.cpu().float().numpy())
            all_labels.append(labels.numpy())
            all_paths.extend(list(paths))

    return (
        np.vstack(all_features).astype(np.float32),
        np.concatenate(all_labels),
        np.array(all_paths, dtype=object),
    )


def extract_features_tta_for_samples(
    model,
    samples,
    device,
    batch_size: int,
    tta_runs: int,
):
    """
    Extract features with TTA (flip augmentation) and average over runs.
    Only used for small classes to stabilize their prototypes.
    """
    loader = DataLoader(
        CroppedDefectDataset(samples, img_size=DINOv2_IMG_SIZE),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    model.eval()
    feats_list = []
    labels_list = []
    paths_list = []

    with torch.no_grad():
        for imgs, labels, paths in tqdm(loader, desc=f"Extract features (TTA x{tta_runs})"):
            imgs = imgs.to(device)
            # Base
            out = model(imgs)
            if isinstance(out, dict):
                feats = out.get("x", out.get("cls_token", list(out.values())[0]))
            else:
                feats = out
            if feats.dim() == 3:
                feats = feats[:, 0]
            feats_sum = feats
            # Augmented runs
            for _ in range(tta_runs - 1):
                aug = _random_flip_batch(imgs)
                out_aug = model(aug)
                if isinstance(out_aug, dict):
                    f_aug = out_aug.get("x", out_aug.get("cls_token", list(out_aug.values())[0]))
                else:
                    f_aug = out_aug
                if f_aug.dim() == 3:
                    f_aug = f_aug[:, 0]
                feats_sum = feats_sum + f_aug
            feats_avg = feats_sum / float(tta_runs)

            feats_list.append(feats_avg.cpu().float().numpy())
            labels_list.append(labels.numpy())
            paths_list.extend(list(paths))

    return (
        np.vstack(feats_list).astype(np.float32),
        np.concatenate(labels_list),
        np.array(paths_list, dtype=object),
    )


def _kmeans_np(x: np.ndarray, k: int, seed: int = 0, iters: int = 25) -> np.ndarray:
    """Simple k-means in numpy. Returns centers (k, D)."""
    n, d = x.shape
    if n == 0:
        return np.zeros((k, d), dtype=np.float32)
    if n <= k:
        # Pad by repeating if too few points
        reps = int(np.ceil(k / n))
        centers = np.tile(x, (reps, 1))[:k].copy()
        return centers.astype(np.float32)

    rng = np.random.default_rng(seed)
    centers = x[rng.choice(n, size=k, replace=False)].copy()
    for _ in range(iters):
        # Assign
        dist2 = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        assign = dist2.argmin(axis=1)
        # Update
        new_centers = []
        for j in range(k):
            pts = x[assign == j]
            if len(pts) == 0:
                new_centers.append(centers[j])
            else:
                new_centers.append(pts.mean(axis=0))
        new_centers = np.stack(new_centers)
        if np.allclose(new_centers, centers, atol=1e-5):
            centers = new_centers
            break
        centers = new_centers
    return centers.astype(np.float32)


def compute_multi_prototypes(features: np.ndarray, labels: np.ndarray, num_classes: int, k: int, seed: int):
    """Compute K prototypes per class using k-means on train features."""
    protos = []
    for c in range(num_classes):
        x = features[labels == c]
        if len(x) == 0:
            protos.append(np.zeros((k, features.shape[1]), dtype=np.float32))
        else:
            kk = min(k, len(x))
            centers = _kmeans_np(x, kk, seed=seed + 31 * c)
            # Pad to k for consistent shape
            if kk < k:
                pad = np.tile(centers, (int(np.ceil(k / kk)), 1))[:k]
                centers = pad
            protos.append(centers)
    return np.stack(protos)  # (C, K, D)


# --- Classifiers ---

def predict_protonet(features: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    """Nearest prototype (ProtoNet): L2 distance."""
    # features: (N, D), prototypes: (C, D)
    diff = features[:, None, :] - prototypes[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    return np.argmin(dist, axis=1)


def predict_cosine(features: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    """Cosine classifier: max cosine similarity."""
    fn = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    pn = prototypes / (np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-8)
    sim = fn @ pn.T
    return np.argmax(sim, axis=1)


def predict_multi_protonet(features: np.ndarray, prototypes_ckd: np.ndarray) -> np.ndarray:
    """Multi-prototype ProtoNet: min L2 distance to any prototype in a class."""
    # features (N,D), prototypes (C,K,D)
    diff = features[:, None, None, :] - prototypes_ckd[None, :, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=3))  # (N,C,K)
    dist_min = dist.min(axis=2)  # (N,C)
    return np.argmin(dist_min, axis=1)


def predict_multi_cosine(features: np.ndarray, prototypes_ckd: np.ndarray) -> np.ndarray:
    """Multi-prototype cosine classifier: max cosine similarity to any prototype in class."""
    fn = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    pn = prototypes_ckd / (np.linalg.norm(prototypes_ckd, axis=2, keepdims=True) + 1e-8)  # (C,K,D)
    # sim (N,C,K)
    sim = np.einsum("nd,ckd->nck", fn, pn)
    sim_max = sim.max(axis=2)  # (N,C)
    return np.argmax(sim_max, axis=1)


def predict_knn(features: np.ndarray, train_features: np.ndarray, train_labels: np.ndarray, k: int = 5, n_classes: int = None) -> np.ndarray:
    """k-NN: majority vote among k nearest neighbors (L2)."""
    if n_classes is None:
        n_classes = int(train_labels.max()) + 1
    diff = features[:, None, :] - train_features[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    nn_idx = np.argsort(dist, axis=1)[:, :k]
    nn_labels = train_labels[nn_idx]
    preds = []
    for i in range(len(features)):
        votes = np.bincount(nn_labels[i].astype(int), minlength=n_classes)
        preds.append(np.argmax(votes))
    return np.array(preds)


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def predict_knn_cosine(
    features: np.ndarray,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    k: int = 5,
    n_classes: int = None,
) -> np.ndarray:
    """k-NN with cosine similarity on L2-normalized features."""
    if n_classes is None:
        n_classes = int(train_labels.max()) + 1
    f = _l2_normalize(features.astype(np.float32))
    t = _l2_normalize(train_features.astype(np.float32))
    # sim: (N, M)
    sim = f @ t.T
    nn_idx = np.argsort(-sim, axis=1)[:, :k]  # top-k highest similarity
    nn_labels = train_labels[nn_idx]
    preds = []
    for i in range(len(f)):
        votes = np.bincount(nn_labels[i].astype(int), minlength=n_classes)
        preds.append(np.argmax(votes))
    return np.array(preds)


def knn_cosine_with_votes(
    features: np.ndarray,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    k: int,
    n_classes: int,
):
    """
    kNN (cosine) returning:
      - preds: (N,)
      - top1_frac: (N,) fraction of neighbors voting for predicted class
      - margin_frac: (N,) (top1_votes - top2_votes) / k
      - top2: (N,) second-best class by votes
    """
    f = _l2_normalize(features.astype(np.float32))
    t = _l2_normalize(train_features.astype(np.float32))
    sim = f @ t.T  # (N,M)
    nn_idx = np.argsort(-sim, axis=1)[:, :k]
    nn_labels = train_labels[nn_idx].astype(int)  # (N,k)

    preds = np.zeros((len(f),), dtype=int)
    top2 = np.zeros((len(f),), dtype=int)
    top1_frac = np.zeros((len(f),), dtype=np.float32)
    margin_frac = np.zeros((len(f),), dtype=np.float32)

    for i in range(len(f)):
        votes = np.bincount(nn_labels[i], minlength=n_classes)
        order = np.argsort(-votes)
        c1 = int(order[0])
        c2 = int(order[1]) if n_classes > 1 else c1
        v1 = int(votes[c1])
        v2 = int(votes[c2])
        preds[i] = c1
        top2[i] = c2
        top1_frac[i] = v1 / float(k)
        margin_frac[i] = (v1 - v2) / float(k)

    return preds, top1_frac, margin_frac, top2


def two_stage_refine_with_rare_prototypes(
    base_preds: np.ndarray,
    low_conf_mask: np.ndarray,
    features: np.ndarray,
    prototypes_ckd: np.ndarray,
    label_names: list,
    rare_names: set,
    min_gain: float,
):
    """
    Stage-2 refinement for low-confidence samples.
    Compare base predicted class vs best rare class using multi-prototype cosine.
    If rare score exceeds base score by `min_gain`, switch to that rare class.
    """
    if not low_conf_mask.any():
        return base_preds, 0

    n = len(label_names)
    rare_ids = [i for i, nme in enumerate(label_names) if nme in rare_names]
    if not rare_ids:
        return base_preds, 0

    # Normalize once
    fn = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    pn = prototypes_ckd / (np.linalg.norm(prototypes_ckd, axis=2, keepdims=True) + 1e-8)  # (C,K,D)

    updated = base_preds.copy()
    changed = 0

    idxs = np.where(low_conf_mask)[0]
    for i in idxs:
        base_c = int(updated[i])
        # base best sim
        base_sim = float(np.einsum("d,kd->k", fn[i], pn[base_c]).max())
        # rare best sim
        rare_best_c = None
        rare_best_sim = -1e9
        for rc in rare_ids:
            s = float(np.einsum("d,kd->k", fn[i], pn[rc]).max())
            if s > rare_best_sim:
                rare_best_sim = s
                rare_best_c = int(rc)
        if rare_best_c is not None and (rare_best_sim - base_sim) >= float(min_gain):
            updated[i] = rare_best_c
            changed += 1

    return updated, changed


def evaluate(preds: np.ndarray, labels: np.ndarray, label_names: list):
    """Compute accuracy and per-class metrics."""
    acc = (preds == labels).mean()
    n_classes = len(label_names)
    per_class_acc = []
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() > 0:
            per_class_acc.append((preds[mask] == labels[mask]).mean())
        else:
            per_class_acc.append(0.0)
    return acc, per_class_acc


def print_confusion_matrix(preds: np.ndarray, labels: np.ndarray, label_names: list):
    """Print confusion matrix: rows=true, cols=pred."""
    n = len(label_names)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    # Header - use short names for readability
    w = 6
    short_names = [name.replace("defect", "d") for name in label_names]
    header = " " * (w + 2) + "".join(f"{s:>{w}}" for s in short_names)
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(header)
    for i, sname in enumerate(short_names):
        row = f"{sname:>{w}}" + " " + "".join(f"{cm[i,j]:{w}d}" for j in range(n))
        print(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=CROPPED_DIR)
    parser.add_argument("--context-dir", type=Path, default=CONTEXT_DIR, help="Context crops root for dual input")
    parser.add_argument("--dual", action="store_true", help="Use dual input: tight crop + context crop (feature concat)")
    parser.add_argument("--method", choices=["protonet", "cosine", "knn", "two_stage"], default="protonet")
    parser.add_argument("--k", type=int, default=5, help="k for kNN")
    parser.add_argument("--proto-k", type=int, default=3, help="K prototypes per class (for protonet/cosine)")
    parser.add_argument("--model-size", default="vitb14", help="DINOv2: vits14, vitb14, vitl14, vitg14")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--save-prototypes", action="store_true", help="Save prototypes to file")
    parser.add_argument("--split-by-image", action="store_true", help="Split by source image (no leakage)")
    parser.add_argument("--stratified", action="store_true", help="Stratify splits by class (recommended for rare classes)")
    parser.add_argument("--tta-small", action="store_true", help="TTA-average features for rare classes (defect3/defect4)")
    parser.add_argument("--tta-runs", type=int, default=8, help="TTA runs for rare classes")
    parser.add_argument("--knn-metric", choices=["l2", "cosine"], default="cosine", help="Distance metric for kNN")
    parser.add_argument("--rare-classes", default="defect3,defect4", help="Comma-separated rare class names for stage-2")
    parser.add_argument("--conf-top1", type=float, default=0.8, help="Stage-1 high confidence if top1 vote fraction >= this")
    parser.add_argument("--conf-margin", type=float, default=0.2, help="Stage-1 high confidence if (top1-top2)/k >= this")
    parser.add_argument("--stage2-min-gain", type=float, default=0.02, help="Switch to rare class if sim gain >= this")
    args = parser.parse_args()

    if not args.data.exists():
        print(f"Data not found: {args.data}")
        return 1

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # DINOv2 uses ops not supported on MPS
    print(f"Device: {device}")

    # Load data
    samples, label_names = load_cropped_defects(args.data)
    n_classes = len(label_names)
    print(f"\nClasses: {label_names}")
    print(f"Total samples: {len(samples)}")

    # Per-class counts
    from collections import Counter
    label_counts = Counter(l for _, l in samples)
    print("Per-class sample count:")
    for i, name in enumerate(label_names):
        print(f"  {name}: {label_counts.get(i, 0)}")

    # Split
    if args.split_by_image and args.stratified:
        train_samples, val_samples, test_samples = split_by_source_image_stratified(
            samples, args.train_ratio, args.val_ratio, args.seed
        )
        print(f"\nSplit by source image (stratified): train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")
    elif args.split_by_image:
        train_samples, val_samples, test_samples = split_by_source_image(
            samples, args.train_ratio, args.val_ratio, args.seed
        )
        print(f"\nSplit by source image: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")
    else:
        indices = np.random.permutation(len(samples))
        n_train = int(len(samples) * args.train_ratio)
        n_val = int(len(samples) * args.val_ratio)
        n_test = len(samples) - n_train - n_val
        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :]
        train_samples = [samples[i] for i in train_idx]
        val_samples = [samples[i] for i in val_idx]
        test_samples = [samples[i] for i in test_idx]
        print(f"\nSplit (random): train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")

    # DINOv2 backbone
    print("\nLoading DINOv2...")
    model, _ = get_dinov2_backbone(args.model_size, device)

    if args.dual:
        if not args.context_dir.exists():
            print(f"Context dir not found: {args.context_dir}")
            print("Please generate it first: python defect_crop.py --save-context")
            return 1
        dataset = lambda s: DualCroppedDefectDataset(s, context_root=args.context_dir, img_size=DINOv2_IMG_SIZE)
    else:
        dataset = lambda s: CroppedDefectDataset(s, img_size=DINOv2_IMG_SIZE)

    train_loader = DataLoader(dataset(train_samples), batch_size=args.batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(dataset(val_samples), batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(dataset(test_samples), batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Extract features
    print("\nExtracting train features...")
    train_feats, train_labels, train_paths = extract_features(model, train_loader, device)
    print("Extracting val features...")
    val_feats, val_labels, val_paths = extract_features(model, val_loader, device)
    print("Extracting test features...")
    test_feats, test_labels, test_paths = extract_features(model, test_loader, device)

    print(f"Feature dim: {train_feats.shape[1]}")

    # Optional: TTA-average features for rare classes (train only)
    if args.tta_small:
        rare_names = {"defect3", "defect4"}
        rare_ids = [i for i, n in enumerate(label_names) if n in rare_names]
        if rare_ids:
            rare_train_samples = [s for s in train_samples if s[1] in rare_ids]
            if rare_train_samples:
                print(f"\nTTA for rare classes {sorted(list(rare_names))}: {len(rare_train_samples)} train crops (runs={args.tta_runs})")
                rare_feats, rare_labels, rare_paths = extract_features_tta_for_samples(
                    model,
                    rare_train_samples,
                    device=device,
                    batch_size=min(args.batch_size, 32),
                    tta_runs=args.tta_runs,
                )
                # Replace features by path
                path_to_idx = {p: i for i, p in enumerate(train_paths.tolist())}
                replaced = 0
                for f, p in zip(rare_feats, rare_paths.tolist()):
                    idx = path_to_idx.get(p)
                    if idx is not None:
                        train_feats[idx] = f
                        replaced += 1
                print(f"Replaced train features with TTA-avg for {replaced} crops")

    # Compute multi-prototypes (from train) for proto/cosine/two_stage stage2
    proto_k = max(1, int(args.proto_k))
    prototypes = compute_multi_prototypes(train_feats, train_labels, n_classes, k=proto_k, seed=args.seed)

    if args.save_prototypes:
        PROTOTYPE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(PROTOTYPE_DIR / "prototypes.npz", prototypes=prototypes, label_names=label_names)
        print(f"Prototypes saved to {PROTOTYPE_DIR / 'prototypes.npz'}")

    # Predict
    rare_names = {x.strip() for x in args.rare_classes.split(",") if x.strip()}

    if args.method == "protonet":
        val_preds = predict_multi_protonet(val_feats, prototypes)
        test_preds = predict_multi_protonet(test_feats, prototypes)
    elif args.method == "cosine":
        val_preds = predict_multi_cosine(val_feats, prototypes)
        test_preds = predict_multi_cosine(test_feats, prototypes)
    elif args.method == "knn":
        if args.knn_metric == "cosine":
            val_preds = predict_knn_cosine(val_feats, train_feats, train_labels, k=args.k, n_classes=n_classes)
            test_preds = predict_knn_cosine(test_feats, train_feats, train_labels, k=args.k, n_classes=n_classes)
        else:
            val_preds = predict_knn(val_feats, train_feats, train_labels, k=args.k, n_classes=n_classes)
            test_preds = predict_knn(test_feats, train_feats, train_labels, k=args.k, n_classes=n_classes)
    else:  # two_stage
        if args.knn_metric != "cosine":
            print("two_stage requires --knn-metric cosine (auto using cosine).")

        # Stage 1: cosine kNN with vote-based confidence
        print(f"\nTwo-stage: stage1=kNN(cosine,k={args.k}), stage2=rare-prototype refine {sorted(list(rare_names))}")
        val_base, val_top1, val_margin, _ = knn_cosine_with_votes(val_feats, train_feats, train_labels, k=args.k, n_classes=n_classes)
        test_base, test_top1, test_margin, _ = knn_cosine_with_votes(test_feats, train_feats, train_labels, k=args.k, n_classes=n_classes)

        val_high = (val_top1 >= args.conf_top1) | (val_margin >= args.conf_margin)
        test_high = (test_top1 >= args.conf_top1) | (test_margin >= args.conf_margin)

        val_low = ~val_high
        test_low = ~test_high

        val_preds, val_changed = two_stage_refine_with_rare_prototypes(
            base_preds=val_base,
            low_conf_mask=val_low,
            features=val_feats,
            prototypes_ckd=prototypes,
            label_names=label_names,
            rare_names=rare_names,
            min_gain=args.stage2_min_gain,
        )
        test_preds, test_changed = two_stage_refine_with_rare_prototypes(
            base_preds=test_base,
            low_conf_mask=test_low,
            features=test_feats,
            prototypes_ckd=prototypes,
            label_names=label_names,
            rare_names=rare_names,
            min_gain=args.stage2_min_gain,
        )

        print(f"Stage1 high-confidence (val):  {val_high.mean()*100:.1f}% | refined on {val_low.sum()} samples | changed {val_changed}")
        print(f"Stage1 high-confidence (test): {test_high.mean()*100:.1f}% | refined on {test_low.sum()} samples | changed {test_changed}")

    # Evaluate
    val_acc, val_per_class = evaluate(val_preds, val_labels, label_names)
    test_acc, test_per_class = evaluate(test_preds, test_labels, label_names)

    print("\n" + "=" * 50)
    print(f"Method: {args.method.upper()}")
    print("=" * 50)
    print(f"Val  Accuracy: {val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print("\nPer-class accuracy (test):")
    for i, (name, acc) in enumerate(zip(label_names, test_per_class)):
        n_test = (test_labels == i).sum()
        print(f"  {name}: {acc:.4f} (n={n_test})")
    print_confusion_matrix(test_preds, test_labels, label_names)
    print("=" * 50)

    return 0


if __name__ == "__main__":
    exit(main())
