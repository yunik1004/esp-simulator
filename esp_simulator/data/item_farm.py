import numpy as np
import torch
from torch.utils.data import Dataset


class ItemFarmDataset(Dataset):
    def __init__(self, file: str) -> None:
        self.data = torch.from_numpy(np.load(file).astype(np.float32))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


class ItemFarmClassificationDataset(Dataset):
    def __init__(self, file_normal: str, file_esp: str) -> None:
        self.dataset_normal = ItemFarmDataset(file_normal)
        self.dataset_esp = ItemFarmDataset(file_esp)

        self.len_normal = len(self.dataset_normal)
        self.len_esp = len(self.dataset_esp)

    def __len__(self) -> int:
        return self.len_normal + self.len_esp

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx < self.len_normal:
            label = 0
            id = idx
            data = self.dataset_normal[id]
        else:
            label = 1
            id = idx - self.len_normal
            data = self.dataset_esp[id]

        output = {
            "data": data,
            "label": label,
            "id": id,
        }
        return output
