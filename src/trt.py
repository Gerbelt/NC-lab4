import os
import time
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import onnxruntime as ort


# ---- 1) Model: ResNet18 for CIFAR-10 (как в твоём обучении)
def build_resnet18_cifar10(num_classes=10):
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


# ---- 2) Export ONNX
def export_onnx(model, onnx_path="model_cifar10.onnx", opset=17):
    model.eval()
    dummy = torch.randn(1, 3, 32, 32)  # одна картинка
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    print(f"[OK] Exported ONNX: {onnx_path}")


# ---- 3) Throughput bench
def throughput(session: ort.InferenceSession, batch=256, steps=200, warmup=50):
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

    ips = total / (t1 - t0)  # images per second
    return ips


def make_session(onnx_path, providers):
    so = ort.SessionOptions()
    # можно включить лог, если нужно доказать что TRT реально используется:
    # so.log_severity_level = 0
    return ort.InferenceSession(onnx_path, sess_options=so, providers=providers)


def main():
    print("onnxruntime:", ort.__version__)
    print("Available providers:", ort.get_available_providers())

    # export
    model = build_resnet18_cifar10()
    load_checkpoint_if_exists(model, "checkpoint.tar", device="cpu")
    export_onnx(model, "model_cifar10.onnx", opset=17)

    # CUDA baseline
    cuda_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess_cuda = make_session("model_cifar10.onnx", cuda_providers)
    ips_cuda = throughput(sess_cuda, batch=256, steps=200, warmup=50)
    print(f"[CUDA] Throughput: {ips_cuda:.1f} images/sec (batch=256)")

    # TensorRT + CUDA fallback
    trt_providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    if "TensorrtExecutionProvider" not in ort.get_available_providers():
        print("[ERROR] TensorrtExecutionProvider is NOT available. Install/configure TensorRT and ensure DLLs are in PATH.")
        return

    sess_trt = make_session("model_cifar10.onnx", trt_providers)
    ips_trt = throughput(sess_trt, batch=256, steps=200, warmup=50)
    print(f"[TensorRT] Throughput: {ips_trt:.1f} images/sec (batch=256)")

    print(f"Speedup (TRT vs CUDA): {ips_trt / ips_cuda:.2f}x")


if __name__ == "__main__":
    main()
