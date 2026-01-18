import platform
import sys
import torch

print("OS:", platform.platform())
print("Python:", sys.version.replace("\n", " "))
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("torch.version.cuda:", torch.version.cuda)
else:
    print("gpu: -")
    print("torch.version.cuda: -")
