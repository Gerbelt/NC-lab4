import os
import time
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models

from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx


# ===== Model: ResNet18 for CIFAR-10 (32x32) =====
def build_resnet18_cifar10(num_classes=10):
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_checkpoint_if_exists(model, path="checkpoint.tar", device="cpu"):
    if not os.path.exists(path):
        print(f"[WARN] checkpoint '{path}' not found -> accuracy will be near random (~10%)")
        return False
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[OK] Loaded checkpoint: {path}")
    return True


def get_test_loader(batch_size=64):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=tfm)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)


@torch.no_grad()
def top1_accuracy(model: nn.Module, loader, n_images=1000):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
        if total >= n_images:
            break
    total = min(total, n_images)
    return correct / total


@torch.no_grad()
def calibrate(model: nn.Module, loader, n_images=256):
    model.eval()
    seen = 0
    for x, _ in loader:
        _ = model(x)
        seen += x.shape[0]
        if seen >= n_images:
            break


def quantize_int8_fx(fp32_model: nn.Module, calib_loader):
    """
    FX Graph Mode PTQ для ResNet:
    prepare_fx -> calibration -> convert_fx
    """
    torch.backends.quantized.engine = "fbgemm"  # для x86 CPU
    fp32_model.eval()

    qconfig_mapping = get_default_qconfig_mapping("fbgemm")

    # пример входа нужен FX для трассировки графа
    example_inputs = (torch.randn(1, 3, 32, 32),)

    prepared = prepare_fx(fp32_model, qconfig_mapping, example_inputs)
    calibrate(prepared, calib_loader, n_images=256)

    int8_model = convert_fx(prepared)
    return int8_model


def write_md(acc_fp32, acc_int8, has_ckpt, path="docs/quantization.md"):
    os.makedirs("docs", exist_ok=True)
    drop_pp = (acc_fp32 - acc_int8) * 100.0

    with open(path, "w", encoding="utf-8") as f:
        f.write("# INT8-точность (Top-1 на 1000 изображениях)\n\n")
        f.write("Сравнение Top-1 точности ResNet18 (CIFAR-10) в float32 и после INT8-квантования.\n\n")

        if not has_ckpt:
            f.write("> ⚠️ В проекте не найден `checkpoint.tar`, поэтому модель использовала случайные веса и точность близка к 10%.\n\n")

        f.write("## Методика\n")
        f.write("- Датасет: CIFAR-10 (test)\n")
        f.write("- Количество изображений: 1000\n")
        f.write("- Метрика: Top-1 accuracy\n")
        f.write("- INT8: PyTorch Post-Training Quantization (FX Graph Mode: prepare_fx → calibration → convert_fx)\n")
        f.write("- Backend: FBGEMM (CPU)\n\n")

        f.write("## Результаты\n")
        f.write("| Вариант | Top-1 (1000) |\n")
        f.write("|---|---:|\n")
        f.write(f"| FP32 | {acc_fp32*100:.2f}% |\n")
        f.write(f"| INT8 | {acc_int8*100:.2f}% |\n\n")

        f.write("## Вывод\n")
        f.write(f"- Падение точности (FP32 → INT8): **{drop_pp:.2f} п.п.**\n")
        f.write("- INT8 снижает разрядность вычислений и может ускорять инференс на CPU. Для ResNet корректнее использовать FX Graph Mode, так как модель содержит residual-сложения.\n")

    print(f"Saved: {path}")


def main():
    print("PyTorch:", torch.__version__)

    loader = get_test_loader(batch_size=64)

    fp32 = build_resnet18_cifar10()
    has_ckpt = load_checkpoint_if_exists(fp32, "checkpoint.tar", device="cpu")

    t0 = time.perf_counter()
    acc_fp32 = top1_accuracy(fp32, loader, n_images=1000)
    t1 = time.perf_counter()
    print(f"[FP32] Top-1 on 1000 images: {acc_fp32*100:.2f}% (time={t1-t0:.2f}s)")

    t0 = time.perf_counter()
    int8 = quantize_int8_fx(fp32, loader)
    t1 = time.perf_counter()
    print(f"[INT8] Built INT8 model (time={t1-t0:.2f}s)")

    t0 = time.perf_counter()
    acc_int8 = top1_accuracy(int8, loader, n_images=1000)
    t1 = time.perf_counter()
    print(f"[INT8] Top-1 on 1000 images: {acc_int8*100:.2f}% (time={t1-t0:.2f}s)")

    write_md(acc_fp32, acc_int8, has_ckpt)


if __name__ == "__main__":
    main()
