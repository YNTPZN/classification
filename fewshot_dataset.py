"""Dataset for few-shot defect classification on cropped images."""

from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

IMG_EXTENSIONS = {".png", ".PNG", ".jpg", ".jpeg", ".JPG", ".JPEG"}
DINOv2_IMG_SIZE = 224  # DINOv2 ViT-B/14 works with 224


def _get_source_stem(path: Path) -> str:
    """Extract source image stem from crop filename: xxx_crop_0.png -> xxx."""
    name = path.stem  # e.g. xxx_crop_0
    if "_crop_" in name:
        return name.rsplit("_crop_", 1)[0]
    return name


def load_cropped_defects(data_root: Path) -> Tuple[List[Tuple[Path, int]], List[str]]:
    """
    Load cropped defect images from folder structure: defect1/, defect2/, ...
    Returns: [(path, label_idx), ...], label_names (e.g. ['defect1', 'defect2', ...])
    """
    folders = sorted([d for d in data_root.iterdir() if d.is_dir() and d.name.startswith("defect")])
    if not folders:
        raise ValueError(f"No defect folders found in {data_root}")

    label_names = [f.name for f in folders]
    name_to_idx = {n: i for i, n in enumerate(label_names)}

    samples = []
    for folder in folders:
        idx = name_to_idx[folder.name]
        for f in folder.iterdir():
            if f.is_file() and f.suffix in IMG_EXTENSIONS:
                samples.append((f, idx))

    return samples, label_names


def split_by_source_image(samples: List[Tuple[Path, int]], train_ratio: float, val_ratio: float, seed: int):
    """
    Split so all crops from the same source image go to the same split.
    Avoids data leakage (same image in train and test).
    """
    from collections import defaultdict
    import numpy as np

    # Group by (label, source_stem)
    groups = defaultdict(list)
    for path, label in samples:
        stem = _get_source_stem(path)
        groups[(label, stem)].append((path, label))

    group_keys = list(groups.keys())
    np.random.seed(seed)
    np.random.shuffle(group_keys)

    n = len(group_keys)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_keys = set(group_keys[:n_train])
    val_keys = set(group_keys[n_train : n_train + n_val])
    test_keys = set(group_keys[n_train + n_val :])

    train_samples, val_samples, test_samples = [], [], []
    for (label, stem), items in groups.items():
        key = (label, stem)
        if key in train_keys:
            train_samples.extend(items)
        elif key in val_keys:
            val_samples.extend(items)
        else:
            test_samples.extend(items)

    return train_samples, val_samples, test_samples


def split_by_source_image_stratified(
    samples: List[Tuple[Path, int]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
):
    """
    Stratified variant of `split_by_source_image`.
    Splits *within each class* by source image groups, so rare classes (e.g. defect3/defect4)
    don't disappear from val/test by chance.
    """
    from collections import defaultdict
    import numpy as np

    # Group by (label, source_stem)
    groups = defaultdict(list)
    for path, label in samples:
        stem = _get_source_stem(path)
        groups[(label, stem)].append((path, label))

    keys_by_label = defaultdict(list)
    for key in groups.keys():
        label, _ = key
        keys_by_label[label].append(key)

    rng = np.random.default_rng(seed)

    train_samples, val_samples, test_samples = [], [], []
    for label, keys in keys_by_label.items():
        keys = list(keys)
        rng.shuffle(keys)

        n = len(keys)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        # If there are enough groups, try to keep at least 1 group in each split.
        if n >= 3:
            n_train = max(1, n_train)
            n_val = max(1, n_val)
            if n_train + n_val >= n:
                n_val = max(1, n - n_train - 1)

        train_keys = set(keys[:n_train])
        val_keys = set(keys[n_train : n_train + n_val])

        for key in keys:
            items = groups[key]
            if key in train_keys:
                train_samples.extend(items)
            elif key in val_keys:
                val_samples.extend(items)
            else:
                test_samples.extend(items)

    return train_samples, val_samples, test_samples


class CroppedDefectDataset(Dataset):
    """Dataset for cropped defect images with DINOv2-style preprocessing."""

    def __init__(self, samples: List[Tuple[Path, int]], img_size: int = DINOv2_IMG_SIZE, transform=None):
        self.samples = samples
        self.img_size = img_size
        if transform is not None:
            self.transform = transform
        else:
            self.transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        return x, label, str(path)


class DualCroppedDefectDataset(Dataset):
    """
    Dataset that returns (tight_crop, context_crop, label, path).
    `samples` uses tight crop paths. `context_root` mirrors folder structure and filenames.
    """

    def __init__(
        self,
        samples: List[Tuple[Path, int]],
        context_root: Path,
        img_size: int = DINOv2_IMG_SIZE,
        transform=None,
    ):
        # Keep only samples that have matching context crops to avoid crashing mid-epoch
        filtered = []
        missing = 0
        for p, y in samples:
            cls_folder = p.parent.name
            ctx_path = context_root / cls_folder / p.name
            if ctx_path.exists():
                filtered.append((p, y))
            else:
                missing += 1
        self.samples = filtered
        self.context_root = context_root
        self.img_size = img_size
        if transform is not None:
            self.transform = transform
        else:
            self.transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        if missing > 0:
            # Keep print lightweight; caller will see this once per dataset construction
            print(f"[DualCroppedDefectDataset] Skipped {missing} samples with missing context crops")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        # path: .../cropped_defects/defectX/xxx_crop_i.png
        cls_folder = path.parent.name
        ctx_path = self.context_root / cls_folder / path.name
        # ctx_path should exist due to filtering in __init__

        img_tight = Image.open(path).convert("RGB")
        img_ctx = Image.open(ctx_path).convert("RGB")
        x1 = self.transform(img_tight)
        x2 = self.transform(img_ctx)
        return (x1, x2), label, str(path)
