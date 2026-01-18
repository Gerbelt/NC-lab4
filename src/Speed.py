import os
import time
import numpy as np

import torch
import torch.nn as nn
from torchvision import models

import onnxruntime as ort

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



def build_resnet18_cifar10(num_classes: int = 10):
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_checkpoint_if_exists(model, path="checkpoint.tar", device="cpu"):
    if not os.path.exists(path):
        print(f"[WARN] checkpoint '{path}' not found -> benchmarking random weights")
        return
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[OK] Loaded checkpoint: {path} (epoch={ckpt.get('epoch')}, best={ckpt.get('best_metric')})")


def export_onnx(model_cpu: nn.Module, onnx_path="model_cifar10.onnx", opset=17):
    model_cpu.eval()
    dummy = torch.randn(1, 3, 32, 32)  # batch=1, CIFAR-10
    torch.onnx.export(
        model_cpu,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    print(f"[OK] Exported ONNX: {onnx_path}")



@torch.no_grad()
def torch_throughput(model: nn.Module, device: torch.device, batch=256, steps=200, warmup=50):
    model.eval()
    x = torch.rand(batch, 3, 32, 32, device=device)

    # warmup
    for _ in range(warmup):
        _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    total = 0
    for _ in range(steps):
        _ = model(x)
        total += batch
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    return total / (t1 - t0)


def ort_session(onnx_path: str, providers, provider_options=None):
    so = ort.SessionOptions()
    if provider_options is None:
        return ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
    return ort.InferenceSession(onnx_path, sess_options=so, providers=providers, provider_options=provider_options)


def ort_throughput(session: ort.InferenceSession, batch=256, steps=200, warmup=50):
    inp_name = session.get_inputs()[0].name
    x = np.random.rand(batch, 3, 32, 32).astype(np.float32)

    # warmup
    for _ in range(warmup):
        session.run(None, {inp_name: x})

    t0 = time.perf_counter()
    total = 0
    for _ in range(steps):
        session.run(None, {inp_name: x})
        total += batch
    t1 = time.perf_counter()

    return total / (t1 - t0)


def save_speedup_plot(pytorch_ips: float, onnx_ips: float, trt_ips: float):
    os.makedirs("docs", exist_ok=True)

    df = pd.DataFrame({
        "Backend": ["PyTorch", "ONNX (CUDA)", "TensorRT (FP16)"],
        "Speed-up": [
            1.0,
            onnx_ips / pytorch_ips,
            trt_ips / pytorch_ips
        ]
    })

    sns.set_theme(style="whitegrid", font_scale=1.2)
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=df, x="Backend", y="Speed-up")

    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.2f}×",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=11
        )

    plt.title("Speed-up: PyTorch vs ONNX vs TensorRT (FP16)")
    plt.ylabel("Speed-up (relative to PyTorch)")
    plt.xlabel("Backend")
    plt.tight_layout()

    out_path = os.path.join("docs", "speedup.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")



def main():
    print("onnxruntime:", ort.__version__)
    print("Available providers:", ort.get_available_providers())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)


    model = build_resnet18_cifar10().to(device)
    load_checkpoint_if_exists(model, "checkpoint.tar", device=device)


    onnx_path = "model_cifar10.onnx"
    model_cpu = build_resnet18_cifar10()
    load_checkpoint_if_exists(model_cpu, "checkpoint.tar", device="cpu")
    export_onnx(model_cpu, onnx_path, opset=17)

    batch = 256
    warmup = 50
    steps = 200


    pytorch_ips = torch_throughput(model, device=device, batch=batch, steps=steps, warmup=warmup)
    print(f"[PyTorch] Throughput: {pytorch_ips:.1f} images/sec (batch={batch})")


    sess_cuda = ort_session(onnx_path, ["CUDAExecutionProvider", "CPUExecutionProvider"])
    onnx_ips = ort_throughput(sess_cuda, batch=batch, steps=steps, warmup=warmup)
    print(f"[ONNX CUDA] Throughput: {onnx_ips:.1f} images/sec (batch={batch})")


    if "TensorrtExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("TensorrtExecutionProvider is not available. Check TensorRT install + PATH.")

    trt_providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]


    trt_opts = {
        "trt_int8_enable": "False",
        "trt_fp16_enable": "True",
    }
    provider_options = [trt_opts, {}, {}]

    sess_trt = ort_session(onnx_path, trt_providers, provider_options=provider_options)
    print("TRT session providers:", sess_trt.get_providers())


    if sess_trt.get_providers()[0] != "TensorrtExecutionProvider":
        raise RuntimeError("TensorRT не активировался (fallback на CPU). Проверь пути без кириллицы и TensorRT DLL.")

    trt_ips = ort_throughput(sess_trt, batch=batch, steps=steps, warmup=warmup)
    print(f"[TensorRT FP16] Throughput: {trt_ips:.1f} images/sec (batch={batch})")

    print(f"Speedup (ONNX CUDA vs PyTorch): {onnx_ips / pytorch_ips:.2f}x")
    print(f"Speedup (TensorRT FP16 vs PyTorch): {trt_ips / pytorch_ips:.2f}x")

    # 6) Plot
    save_speedup_plot(pytorch_ips, onnx_ips, trt_ips)


if __name__ == "__main__":
    main()
