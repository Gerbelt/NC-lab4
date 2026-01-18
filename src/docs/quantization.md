# INT8-точность (Top-1 на 1000 изображениях)

Сравнение Top-1 точности ResNet18 (CIFAR-10) в float32 и после INT8-квантования.

## Методика
- Датасет: CIFAR-10 (test)
- Количество изображений: 1000
- Метрика: Top-1 accuracy
- INT8: PyTorch Post-Training Quantization (FX Graph Mode: prepare_fx → calibration → convert_fx)
- Backend: FBGEMM (CPU)

## Результаты
| Вариант | Top-1 (1000) |
|---|---:|
| FP32 | 83.70% |
| INT8 | 83.60% |

## Вывод
- Падение точности (FP32 → INT8): **0.10 п.п.**
