#!/usr/bin/env python3
"""
train_defect_vits14_sce.py
---------------------------------
Fine-tune DINOv2 ViT-S/14 for single-head defect classification on a small, noisy dataset.
Features:
  - Symmetric Cross-Entropy (SCE) loss (robust to label noise)
  - MixUp / CutMix (timm) with soft targets
  - Class-balanced sampling (optional)
  - Grouped split by "group" column (e.g., report/building), else stratified
  - Cosine LR schedule with warmup
  - Early stopping on macro-F1
  - Saves: best checkpoint, metrics CSV, confusion matrix CSV, label-to-index json

CSV format (minimum):
  path,label[,group]

Example:
  path,label,group
  /data/imgs/17-034_page5_img1.png,Debris,17-034
  /data/imgs/17-034_page6_img2.png,Drain Screen Clogged,17-034
  ...

Usage:
  python train_defect_vits14_sce.py --csv dataset_merged_with_pseudolabels.csv \
      --outdir runs/vits14_sce --img-size 288 --epochs 80 --mixup 0.1 --cutmix 0.1 --balanced-sampler

Requirements:
  pip install torch torchvision timm pandas scikit-learn numpy pillow
"""

import os
import json
import math
import random
import argparse
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import timm
from timm.data.mixup import Mixup
from timm.data.random_erasing import RandomErasing

from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.metrics import f1_score, classification_report, confusion_matrix


# -------------------------- Utils --------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(labels.long(), num_classes=num_classes).float()


# -------------------------- Dataset --------------------------

class CSVImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_root: str, img_size: int, is_train: bool):
        self.df = df.reset_index(drop=True)
        self.img_root = img_root
        self.is_train = is_train
        self.img_size = img_size

        # Basic transforms (augments handled by Mixup; keep spatial aug light here)
        import torchvision.transforms as T
        if is_train:
            self.tf = T.Compose([
                T.Resize(int(img_size * 1.15)),
                T.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomPerspective(distortion_scale=0.05, p=0.2),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.05, hue=0.02),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            self.tf = T.Compose([
                T.Resize(int(img_size * 1.05)),
                T.CenterCrop(img_size),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = row["path"]
        if not os.path.isabs(path):
            path = os.path.join(self.img_root, path)
        img = Image.open(path).convert("RGB")
        img = self.tf(img)
        label = int(row["y"])
        return img, label


# -------------------------- Loss: SCE --------------------------

class SCELoss(nn.Module):
    """
    Symmetric Cross Entropy:
      SCE = alpha * CE(t, p) + beta * RCE(t, p)
      CE  = -sum_i t_i * log p_i
      RCE = -sum_i p_i * log t_i   (with t_i clipped to [eps,1] to avoid -inf)

    - Accepts hard labels (B,) or soft targets (B,C) (e.g., from Mixup/CutMix).
    - For hard labels, optional label smoothing is applied to build soft targets.
    """
    def __init__(self, num_classes: int, alpha: float = 0.1, beta: float = 1.0, label_smooth: float = 0.0, eps: float = 1e-4):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.label_smooth = label_smooth
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: [B,C]; target: [B] (int) or [B,C] (prob)
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        if target.dim() == 1:
            t = one_hot(target, logits.size(1))
            if self.label_smooth > 0:
                t = t * (1.0 - self.label_smooth) + self.label_smooth / self.num_classes
        else:
            t = target
        t = t.clamp(self.eps, 1.0)

        ce = -(t * log_probs).sum(dim=1).mean()
        rce = -(probs * torch.log(t)).sum(dim=1).mean()

        return self.alpha * ce + self.beta * rce


# -------------------------- Model --------------------------

def create_model(num_classes: int, drop_path: float = 0.1):
    # DINOv2 ViT-S/14 backbone with classification head
    model = timm.create_model(
        "vit_small_patch14_dinov2.lvd142m",
        pretrained=True,
        num_classes=num_classes,
        drop_path_rate=drop_path,
    )
    return model


# -------------------------- Sampler --------------------------

def make_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    # Inverse frequency weights
    unique, counts = np.unique(labels, return_counts=True)
    freq = {int(k): int(v) for k, v in zip(unique, counts)}
    weights = np.array([1.0 / freq[int(y)] for y in labels], dtype=np.float32)
    sampler = WeightedRandomSampler(weights.tolist(), num_samples=len(labels), replacement=True)
    return sampler


# -------------------------- Train / Eval --------------------------

@torch.no_grad()
def evaluate(model, loader, device, num_classes: int) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    model.eval()
    all_logits = []
    all_targets = []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(imgs)
        all_logits.append(logits.cpu())
        all_targets.append(labels.cpu())
    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0).numpy()
    probs = F.softmax(logits, dim=1).numpy()
    preds = probs.argmax(axis=1)
    top1 = (preds == targets).mean().item()
    # Top-3
    top3 = np.mean([t in np.argpartition(p, -3)[-3:] for p, t in zip(probs, targets)]).item()
    macro_f1 = f1_score(targets, preds, average="macro")
    return top1, top3, macro_f1, preds, targets


def train(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.outdir, exist_ok=True)

    # --------- Load CSV ---------
    df = pd.read_csv(args.csv)
    if "label" not in df.columns or "path" not in df.columns:
        raise SystemExit("CSV must contain columns: path,label[,group]")

    # Map labels -> y (ints)
    classes = sorted(df["label"].dropna().unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    df = df[df["label"].astype(str).str.len() > 0].copy()  # drop unlabeled rows
    df["y"] = df["label"].map(class_to_idx).astype(int)

    # Save label mapping
    with open(os.path.join(args.outdir, "label_to_index.json"), "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, ensure_ascii=False, indent=2)

    # --------- Split (grouped or stratified) ---------
    if args.use_groups and "group" in df.columns:
        gss = GroupShuffleSplit(n_splits=1, test_size=args.val_frac, random_state=args.seed)
        train_idx, val_idx = next(gss.split(df, groups=df["group"]))
    else:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=args.val_frac, random_state=args.seed)
        train_idx, val_idx = next(sss.split(df, df["y"]))

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val   = df.iloc[val_idx].reset_index(drop=True)

    num_classes = len(classes)
    print(f"Classes: {num_classes}")
    print(f"Train: {len(df_train)} | Val: {len(df_val)}")

    # --------- Datasets / Loaders ---------
    ds_train = CSVImageDataset(df_train, args.img_root, args.img_size, is_train=True)
    ds_val   = CSVImageDataset(df_val,   args.img_root, args.img_size, is_train=False)

    if args.balanced_sampler:
        sampler = make_weighted_sampler(df_train["y"].values)
        train_loader = DataLoader(ds_train, batch_size=args.batch_size, sampler=sampler,
                                  num_workers=args.num_workers, pin_memory=True, drop_last=True)
    else:
        train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True, drop_last=True)

    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # --------- Model / Optim / Sched ---------
    model = create_model(num_classes=num_classes, drop_path=args.drop_path).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Cosine schedule with warmup
    def lr_lambda(step):
        total_steps = max(1, int(len(train_loader) * args.epochs))
        warmup_steps = int(args.warmup_epochs * len(train_loader))
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Loss (SCE) — supports hard or soft targets
    criterion = SCELoss(num_classes=num_classes, alpha=args.sce_alpha, beta=args.sce_beta,
                        label_smooth=args.label_smooth, eps=1e-4)

    # Mixup/CutMix wrapper (applies in collate_fn style)
    mixup_fn = None
    if args.mixup > 0.0 or args.cutmix > 0.0:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            label_smoothing=0.0,  # we already handle smoothing in SCE as needed
            num_classes=num_classes
        )

    # For logging
    hist = []
    best_f1 = -1.0
    best_path = os.path.join(args.outdir, "best.pth")
    epochs_no_improve = 0

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_seen = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if mixup_fn is not None:
                imgs, targets = mixup_fn(imgs, labels)  # targets: soft (B,C)
            else:
                targets = labels  # hard (B,)

            logits = model(imgs)
            loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            bs = imgs.size(0)
            running_loss += loss.item() * bs
            n_seen += bs
            global_step += 1

        train_loss = running_loss / max(1, n_seen)

        # ---- Eval ----
        top1, top3, macro_f1, preds, targets_np = evaluate(model, val_loader, device, num_classes)
        print(f"Epoch {epoch:03d} | train_loss {train_loss:.4f} | val_top1 {top1*100:.2f}% | val_top3 {top3*100:.2f}% | macroF1 {macro_f1*100:.2f}%")

        hist.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_top1": float(top1),
            "val_top3": float(top3),
            "val_macro_f1": float(macro_f1),
            "lr": float(optimizer.param_groups[0]["lr"]),
        })
        pd.DataFrame(hist).to_csv(os.path.join(args.outdir, "training_log.csv"), index=False)

        # Early stopping
        if macro_f1 > best_f1 + 1e-6:
            best_f1 = macro_f1
            epochs_no_improve = 0
            torch.save({"model": model.state_dict(),
                        "classes": classes,
                        "args": vars(args)}, best_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping after {epoch} epochs. Best macro-F1: {best_f1*100:.2f}%")
                break

    print(f"Best macro-F1: {best_f1*100:.2f}%  | checkpoint: {best_path}")

    # Final eval + confusion matrix on val split (best model)
    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    top1, top3, macro_f1, preds, targets_np = evaluate(model, val_loader, device, num_classes)

    # Reports
    rep = classification_report(targets_np, preds, target_names=classes, digits=4, zero_division=0)
    print("\nValidation report:\n", rep)
    with open(os.path.join(args.outdir, "val_classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(rep)

    cm = confusion_matrix(targets_np, preds, labels=list(range(num_classes)))
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    cm_df.to_csv(os.path.join(args.outdir, "val_confusion_matrix.csv"))

    print("Saved:")
    print(" - best model:", best_path)
    print(" - training log:", os.path.join(args.outdir, "training_log.csv"))
    print(" - classification report:", os.path.join(args.outdir, "val_classification_report.txt"))
    print(" - confusion matrix CSV:", os.path.join(args.outdir, "val_confusion_matrix.csv"))
    print(" - label mapping:", os.path.join(args.outdir, "label_to_index.json"))


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--csv", type=str, required=True, help="CSV with columns: path,label[,group]")
    p.add_argument("--img-root", type=str, default=".", help="Prefix added if CSV paths are relative")
    p.add_argument("--outdir", type=str, default="runs/vits14_sce", help="Output directory")
    p.add_argument("--img-size", type=int, default=288, help="Image size (square)")
    p.add_argument("--epochs", type=int, default=80, help="Max epochs")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size")
    p.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    p.add_argument("--val-frac", type=float, default=0.2, help="Validation fraction if splitting")
    p.add_argument("--use-groups", action="store_true", help="Use 'group' column for grouped split")
    p.add_argument("--balanced-sampler", action="store_true", help="Use class-balanced sampler for training")
    p.add_argument("--mixup", type=float, default=0.1, help="MixUp alpha (0 to disable)")
    p.add_argument("--cutmix", type=float, default=0.1, help="CutMix alpha (0 to disable)")
    p.add_argument("--sce-alpha", type=float, default=0.1, help="SCE alpha")
    p.add_argument("--sce-beta", type=float, default=1.0, help="SCE beta")
    p.add_argument("--label-smooth", type=float, default=0.0, help="Label smoothing inside SCE for hard labels")
    p.add_argument("--drop-path", type=float, default=0.1, help="Stochastic depth rate")
    p.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    p.add_argument("--warmup-epochs", type=float, default=3.0, help="Warmup epochs (fractional OK)")
    p.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    p.add_argument("--patience", type=int, default=10, help="Early stopping patience (epochs)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
