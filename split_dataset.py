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

By default, this script performs *stratified* splitting per class folder:
  - Each top-level class directory under Dataset/ is split independently.
  - This ensures rare classes (e.g. defect3/defect4) will still appear in test.
"""

import argparse
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, default=Path(__file__).resolve().parent.parent / "Dataset")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-test-per-class", type=int, default=1)
    p.add_argument("--img-exts", type=str, default=".png,.jpg,.jpeg,.PNG,.JPG,.JPEG")
    p.add_argument("--out-train", type=Path, default=Path(__file__).resolve().parent / "output" / "train_images.txt")
    p.add_argument("--out-test", type=Path, default=Path(__file__).resolve().parent / "output" / "test_images.txt")
    args = p.parse_args()

    root = args.dataset_root
    if not root.exists():
        print(f"Dataset root not found: {root}")
        return 1

    img_exts = {e.strip() for e in args.img_exts.split(",") if e.strip()}

    # Stratified split per class folder
    rng = np.random.default_rng(args.seed)
    all_train = []
    all_test = []
    per_class = []

    class_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if not class_dirs:
        print(f"No class directories found under {root}")
        return 1

    for cls_dir in class_dirs:
        cls_name = cls_dir.name
        rel_paths = []
        for p in sorted(cls_dir.iterdir()):
            if p.is_file() and p.suffix in img_exts:
                rel_paths.append(p.relative_to(root))

        if not rel_paths:
            continue

        idx = np.arange(len(rel_paths))
        rng.shuffle(idx)

        n = len(rel_paths)
        n_train = int(n * args.train_ratio)
        n_test = n - n_train

        # Ensure test is non-empty for rare classes when possible
        if n >= 2 and n_test == 0 and args.min_test_per_class >= 1:
            n_train = n - 1
            n_test = 1
        if n >= 2 and n_train == 0:
            n_train = 1
            n_test = n - 1

        train_idx = idx[:n_train]
        test_idx = idx[n_train:]

        train_paths = [rel_paths[i] for i in train_idx]
        test_paths = [rel_paths[i] for i in test_idx]

        all_train.extend(train_paths)
        all_test.extend(test_paths)
        per_class.append((cls_name, n, len(train_paths), len(test_paths)))

    if not all_train and not all_test:
        print(f"No images found under {root}")
        return 1

    out_dir = args.out_train.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with args.out_train.open("w") as f:
        for rel in sorted(all_train):
            f.write(str(rel) + "\n")

    with args.out_test.open("w") as f:
        for rel in sorted(all_test):
            f.write(str(rel) + "\n")

    total_n = len(all_train) + len(all_test)
    print(f"Total images: {total_n}")
    print(f"Train: {len(all_train)} -> {args.out_train}")
    print(f"Test:  {len(all_test)} -> {args.out_test}")
    print("Per-class split:")
    for cls_name, n, n_tr, n_te in per_class:
        print(f"  {cls_name}: {n} total -> {n_tr} train, {n_te} test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

