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



def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True  # speed
    torch.backends.cudnn.deterministic = False



class AlbumentationsDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, transform=None):
        self.base = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]  # img is PIL
        img = np.array(img)          # HWC, uint8
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        return img, label



def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    correct = 0
    total = 0
    running_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=(device.type == "cuda")):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

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
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        running_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return running_loss / total, correct / total


def main():
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # CIFAR-10 mean/std
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    train_tfms = A.Compose([
        A.RandomCrop(32, 32, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.10, rotate_limit=15, p=0.7),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.08, p=0.7),
        A.CoarseDropout(max_holes=1, max_height=8, max_width=8, min_holes=1, min_height=8, min_width=8, p=0.5),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    val_tfms = A.Compose([
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    # Datasets
    base_train = datasets.CIFAR10(root="./data", train=True, download=True)
    base_test  = datasets.CIFAR10(root="./data", train=False, download=True)

    train_ds = AlbumentationsDataset(base_train, transform=train_tfms)
    val_ds   = AlbumentationsDataset(base_test,  transform=val_tfms)

    # Loaders
    batch_size = 256 
    num_workers = 4

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    epochs = 10
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.25,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.2,
        div_factor=10.0,
        final_div_factor=100.0,
        anneal_strategy="cos",
    )

    scaler = GradScaler(enabled=(device.type == "cuda"))

    train_acc_hist = []
    val_acc_hist = []
    train_loss_hist = []
    val_loss_hist = []

    best_val = 0.0

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)

        break

    def train_one_epoch_with_sched(model, loader, criterion, optimizer, scaler, scheduler, device):
        model.train()
        correct = 0
        total = 0
        running_loss = 0.0

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=(device.type == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        return running_loss / total, correct / total
    train_acc_hist.clear()
    val_acc_hist.clear()
    train_loss_hist.clear()
    val_loss_hist.clear()

    best_val = 0.0
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch_with_sched(
            model, train_loader, criterion, optimizer, scaler, scheduler, device
        )
        va_loss, va_acc = eval_one_epoch(model, val_loader, criterion, device)

        train_loss_hist.append(tr_loss)
        val_loss_hist.append(va_loss)
        train_acc_hist.append(tr_acc)
        val_acc_hist.append(va_acc)

        if va_acc > best_val:
            best_val = va_acc

        print(f"Epoch {epoch:02d}/{epochs} | "
              f"train acc: {tr_acc*100:.2f}% loss: {tr_loss:.4f} | "
              f"val acc: {va_acc*100:.2f}% loss: {va_loss:.4f}")

    # Plot accuracy
    plt.figure()
    plt.plot(range(1, epochs + 1), [a * 100 for a in train_acc_hist], label="train_acc")
    plt.plot(range(1, epochs + 1), [a * 100 for a in val_acc_hist], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"CIFAR-10 Accuracy (best val: {best_val*100:.2f}%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("accuracy_plot.png", dpi=150)
    print("Saved: accuracy_plot.png")


if __name__ == "__main__":
    main()
