"""Dataset loader for defect screening - good vs defect classification."""

import os
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


def get_image_paths(data_root: Path, folder_name: str) -> List[Path]:
    """Get all image paths from a folder."""
    extensions = {".png", ".PNG", ".jpg", ".jpeg", ".JPG", ".JPEG"}
    folder = data_root / folder_name
    if not folder.exists():
        return []
    paths = []
    for f in folder.iterdir():
        if f.is_file() and f.suffix in extensions:
            paths.append(f)
    return paths


def load_dataset(
    data_root: Path,
    good_folder: str = "good",
    defect_prefix: str = "defect",
) -> Tuple[List[Tuple[Path, int]], List[str]]:
    """
    Load dataset from good and defect folders.
    Returns: [(path, label), ...], label_names
    label 0 = good (normal), label 1 = defect (abnormal)
    """
    all_samples = []
    
    # Good images (label 0)
    good_paths = get_image_paths(data_root, good_folder)
    for p in good_paths:
        all_samples.append((p, 0))
    
    # Defect images (label 1) - all folders starting with defect_prefix
    for item in sorted(data_root.iterdir()):
        if item.is_dir() and item.name.startswith(defect_prefix):
            defect_paths = get_image_paths(data_root, item.name)
            for p in defect_paths:
                all_samples.append((p, 1))
    
    label_names = ["good", "defect"]
    return all_samples, label_names


class DefectDataset(Dataset):
    """PyTorch Dataset for defect screening."""
    
    def __init__(
        self,
        samples: List[Tuple[Path, int]],
        transform=None,
        img_size: int = 224,
        is_training: bool = True,
    ):
        self.samples = samples
        self.img_size = img_size
        self.is_training = is_training
        
        if transform is not None:
            self.transform = transform
        else:
            if is_training:
                self.transform = T.Compose([
                    T.Resize((img_size, img_size)),
                    T.RandomHorizontalFlip(),
                    T.RandomVerticalFlip(),
                    T.RandomRotation(15),
                    T.ColorJitter(brightness=0.2, contrast=0.2),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])
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
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load {path}: {e}")
        
        img_tensor = self.transform(img)
        return img_tensor, label, str(path)
