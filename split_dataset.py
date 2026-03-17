#!/usr/bin/env python3
"""
Create a global train/test split on original images under Dataset/.

Assumes structure:
  Dataset/
    good/
    defect1/
    defect2/
    ...

Writes two text files (one path per line, relative to Dataset/):
  output/train_images.txt
  output/test_images.txt
"""

import argparse
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, default=Path(__file__).resolve().parent.parent / "Dataset")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-train", type=Path, default=Path(__file__).resolve().parent / "output" / "train_images.txt")
    p.add_argument("--out-test", type=Path, default=Path(__file__).resolve().parent / "output" / "test_images.txt")
    args = p.parse_args()

    root = args.dataset_root
    if not root.exists():
        print(f"Dataset root not found: {root}")
        return 1

    # Collect all image paths under Dataset/* (good and defect*)
    img_exts = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}
    rel_paths = []
    for cls_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        for p in sorted(cls_dir.iterdir()):
            if p.is_file() and p.suffix in img_exts:
                rel_paths.append(p.relative_to(root))

    if not rel_paths:
        print(f"No images found under {root}")
        return 1

    N = len(rel_paths)
    idx = np.arange(N)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(idx)
    n_train = int(N * args.train_ratio)
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]

    out_dir = args.out_train.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with args.out_train.open("w") as f:
        for i in train_idx:
            f.write(str(rel_paths[i]) + "\n")

    with args.out_test.open("w") as f:
        for i in test_idx:
            f.write(str(rel_paths[i]) + "\n")

    print(f"Total images: {N}")
    print(f"Train: {len(train_idx)} -> {args.out_train}")
    print(f"Test:  {len(test_idx)} -> {args.out_test}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

