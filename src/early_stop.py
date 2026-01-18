import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torchvision import datasets, models
import matplotlib.pyplot as plt

import albumentations as A
from albumentations.pytorch import ToTensorV2


# =========================
# Reproducibility
# =========================
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Speedup (OK for deep learning)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


# =========================
# Albumentations wrapper
# =========================
class AlbumentationsDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, transform=None):
        self.base = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]          # PIL image
        img = np.array(img)                  # to HWC uint8
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        return img, label


# =========================
# Early Stopping
# =========================
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.0, mode="max"):
        """
        mode:
          - "max": metric should increase (accuracy)
          - "min": metric should decrease (loss)
        """
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode

        self.best = None
        self.bad_epochs = 0

    def _is_improvement(self, current: float) -> bool:
        if self.best is None:
            return True

        if self.mode == "max":
            return current > (self.best + self.min_delta)
        else:
            return current < (self.best - self.min_delta)

    def step(self, current: float) -> bool:
        """Return True if improved, else False."""
        if self._is_improvement(current):
            self.best = current
            self.bad_epochs = 0
            return True
        else:
            self.bad_epochs += 1
            return False

    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience


# =========================
# Checkpoint save/load
# =========================
def save_checkpoint(path: str, model: nn.Module, optimizer: optim.Optimizer,
                    epoch: int, best_metric: float,
                    scaler: GradScaler | None = None,
                    scheduler: object | None = None):
    ckpt = {
        "epoch": epoch,
        "best_metric": best_metric,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if scaler is not None:
        ckpt["scaler_state_dict"] = scaler.state_dict()
    if scheduler is not None:
        # Some schedulers have state_dict
        try:
            ckpt["scheduler_state_dict"] = scheduler.state_dict()
        except Exception:
            pass

    torch.save(ckpt, path)


def load_checkpoint(path: str, model: nn.Module, optimizer: optim.Optimizer,
                    scaler: GradScaler | None = None,
                    scheduler: object | None = None,
                    map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scaler is not None and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in ckpt:
        try:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        except Exception:
            pass

    return ckpt["epoch"], ckpt.get("best_metric", None)


# =========================
# Train / Eval
# =========================
def train_one_epoch(model, loader, criterion, optimizer, scaler, scheduler, device):
    model.train()
    correct = 0
    total = 0
    running_loss = 0.0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=(device.type == "cuda")):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        running_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return running_loss / total, correct / total


# =========================
# Main
# =========================
def main():
    seed_everything(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # CIFAR-10 mean/std
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    # Albumentations transforms
    train_tfms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.10, rotate_limit=15, p=0.7),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.08, p=0.7),
        A.CoarseDropout(max_holes=1, max_height=8, max_width=8, p=0.5),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    test_tfms = A.Compose([
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    # Datasets
    base_train = datasets.CIFAR10(root="./data", train=True, download=True)
    base_test  = datasets.CIFAR10(root="./data", train=False, download=True)

    train_ds = AlbumentationsDataset(base_train, transform=train_tfms)
    test_ds  = AlbumentationsDataset(base_test, transform=test_tfms)

    # Loaders
    batch_size = 256
    num_workers = 4

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=512, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    # Model: ResNet18 (хорошо подходит для CIFAR-10)
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)

    # Loss / Optimizer / Scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    epochs = 50  # можно поставить больше, но early stopping остановит раньше
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.25,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.2,
        div_factor=10.0,
        final_div_factor=100.0,
        anneal_strategy="cos",
    )

    scaler = GradScaler(enabled=(device.type == "cuda"))

    # Early stopping по test accuracy
    early = EarlyStopping(patience=5, min_delta=0.001, mode="max")
    checkpoint_path = "checkpoint.tar"

    train_acc_hist = []
    test_acc_hist = []

    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, scheduler, device
        )
        te_loss, te_acc = eval_one_epoch(
            model, test_loader, criterion, device
        )

        train_acc_hist.append(tr_acc)
        test_acc_hist.append(te_acc)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_acc={tr_acc*100:.2f}% train_loss={tr_loss:.4f} | "
            f"test_acc={te_acc*100:.2f}% test_loss={te_loss:.4f}"
        )

        improved = early.step(te_acc)
        if improved:
            best_acc = te_acc
            save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_metric=best_acc,
                scaler=scaler,
                scheduler=scheduler
            )
            print(f"  ✓ Улучшение! Сохранён {checkpoint_path} (best_acc={best_acc*100:.2f}%)")
        else:
            print(f"  ✗ Нет улучшения {early.bad_epochs}/{early.patience}")

        if early.should_stop():
            print(f"Early stopping: нет улучшений {early.patience} эпох подряд. Останавливаемся.")
            break

    # Plot accuracy
    plt.figure()
    plt.plot(range(1, len(train_acc_hist) + 1), [a * 100 for a in train_acc_hist], label="train_acc")
    plt.plot(range(1, len(test_acc_hist) + 1), [a * 100 for a in test_acc_hist], label="test_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"CIFAR-10 Accuracy (best test: {best_acc*100:.2f}%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("accuracy_plot.png", dpi=150)
    print("Saved: accuracy_plot.png")

    print(f"Best accuracy: {best_acc*100:.2f}%")
    print(f"Checkpoint saved to: {os.path.abspath(checkpoint_path)}")



if __name__ == "__main__":
    main()
