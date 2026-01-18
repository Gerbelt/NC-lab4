import torch
print("Версия PyTorch:", torch.__version__)
print("Версия CUDA в PyTorch:", torch.version.cuda)

# Проверка CUDNN (библиотека для глубокого обучения)
print("CUDNN доступен:", torch.backends.cudnn.enabled)
print("Версия CUDNN:", torch.backends.cudnn.version())