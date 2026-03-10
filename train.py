#!/usr/bin/env python3
"""Train defect classifier on good vs defect images."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from config import (
    DATA_ROOT,
    GOOD_FOLDER,
    DEFECT_PREFIX,
    OUTPUT_DIR,
    MODEL_SAVE_PATH,
    MODEL_NAME,
    IMG_SIZE,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    VAL_SPLIT,
    TEST_SPLIT,
    RANDOM_SEED,
)
from dataset import load_dataset, DefectDataset
from model import build_classifier


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels, _ in tqdm(loader, desc="Train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (out.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for imgs, labels, _ in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            loss = criterion(out, labels)
            total_loss += loss.item()
            correct += (out.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_ROOT)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--output", type=Path, default=MODEL_SAVE_PATH)
    args = parser.parse_args()
    
    set_seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    samples, label_names = load_dataset(args.data, GOOD_FOLDER, DEFECT_PREFIX)
    print(f"Total samples: {len(samples)}")
    n_good = sum(1 for _, l in samples if l == 0)
    n_defect = sum(1 for _, l in samples if l == 1)
    print(f"  Good: {n_good}, Defect: {n_defect}")
    
    # Split
    n_val = int(len(samples) * VAL_SPLIT)
    n_test = int(len(samples) * TEST_SPLIT)
    n_train = len(samples) - n_val - n_test
    train_s, val_s, test_s = random_split(samples, [n_train, n_val, n_test])
    
    train_dataset = DefectDataset(
        [samples[i] for i in train_s.indices],
        img_size=IMG_SIZE,
        is_training=True,
    )
    val_dataset = DefectDataset(
        [samples[i] for i in val_s.indices],
        img_size=IMG_SIZE,
        is_training=False,
    )
    
    num_workers = 4 if torch.cuda.is_available() else 0  # MPS/macOS may have issues with multiprocessing
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    
    model = build_classifier(MODEL_NAME, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    # Class weights for imbalanced data (more good than defect)
    n_total = n_good + n_defect
    weights = torch.tensor([1.0 / (n_good / n_total), 1.0 / (n_defect / n_total)], dtype=torch.float32).to(device)
    weights = weights / weights.sum()
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_acc = 0
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_name": MODEL_NAME,
                "img_size": IMG_SIZE,
                "label_names": label_names,
            }, args.output)
            print(f"  -> Saved best model (acc={val_acc:.4f})")
    
    print(f"\nTraining done. Best val acc: {best_acc:.4f}")
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
