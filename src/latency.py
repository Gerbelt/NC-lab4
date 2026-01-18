import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, models
import matplotlib.pyplot as plt

import albumentations as A
from albumentations.pytorch import ToTensorV2


class AlbumentationsDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, transform=None):
        self.base = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]  # PIL
        img = np.array(img)          # HWC uint8
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, label


def build_resnet18_cifar10(num_classes=10):
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_checkpoint_if_exists(model, path="checkpoint.tar", device="cpu"):
    if not os.path.exists(path):
        print(f"[INFO] Checkpoint '{path}' not found. Measuring latency with random weights.")
        return
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[INFO] Loaded checkpoint '{path}' (epoch={ckpt.get('epoch')}, best_metric={ckpt.get('best_metric')}).")


@torch.no_grad()
def measure_latency_100(model, x, device, runs=100, warmup=20):
    model.eval()
    x = x.to(device)

    # Warmup
    for _ in range(warmup):
        _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()
        lat_ms = []
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        for _ in range(runs):
            starter.record()
            _ = model(x)
            ender.record()
            torch.cuda.synchronize()
            lat_ms.append(starter.elapsed_time(ender))  # ms

        return np.array(lat_ms, dtype=np.float32)

    import time
    lat_ms = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = model(x)
        t1 = time.perf_counter()
        lat_ms.append((t1 - t0) * 1000.0)
    return np.array(lat_ms, dtype=np.float32)


def save_latency_plot(lat_ms, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    mean = float(lat_ms.mean())
    p50 = float(np.percentile(lat_ms, 50))
    p95 = float(np.percentile(lat_ms, 95))

    plt.figure()
    plt.plot(range(1, len(lat_ms) + 1), lat_ms, marker="o", linestyle="-")
    plt.xlabel("Прогон")
    plt.ylabel("Latency (ms)")
    plt.title(f"Latency-100 (mean={mean:.3f}ms, p50={p50:.3f}ms, p95={p95:.3f}ms)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


def main():
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    tfm = A.Compose([
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    base_test = datasets.CIFAR10(root="./data", train=False, download=True)
    test_ds = AlbumentationsDataset(base_test, transform=tfm)

    x, y = test_ds[0]
    x = x.unsqueeze(0)  # (1, 3, 32, 32)

    model = build_resnet18_cifar10().to(device)

    load_checkpoint_if_exists(model, path="checkpoint.tar", device=device)

    lat_ms = measure_latency_100(model, x, device, runs=100, warmup=20)

    print(f"Latency stats (ms): mean={lat_ms.mean():.3f}, p50={np.percentile(lat_ms,50):.3f}, p95={np.percentile(lat_ms,95):.3f}")
    save_latency_plot(lat_ms, out_path=os.path.join("docs", "latency.png"))


if __name__ == "__main__":
    main()
