import numpy as np

class AlbumentationsDataset:

    def __init__(self, base_dataset, transform):
        self.base = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]  # img = PIL.Image
        img = np.array(img)          # -> HWC, uint8
        out = self.transform(image=img)
        x = out["image"]
        return x, label
