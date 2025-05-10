import random

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
    def __init__(
        self, file_normal: str, file_esp: str, contam_ratio: float = 0.0
    ) -> None:
        self.dataset_normal = ItemFarmDataset(file_normal)
        self.dataset_esp = ItemFarmDataset(file_esp)
        self.contam_ratio = contam_ratio

        self.len_normal = len(self.dataset_normal)
        self.len_esp = len(self.dataset_esp)

        num_flips_normal = int(self.len_normal * self.contam_ratio)
        num_flips_esp = int(self.len_esp * self.contam_ratio)

        normal_indices = list(range(self.len_normal))
        random.shuffle(normal_indices)
        self.flip_normal_indices = set(normal_indices[:num_flips_normal])

        esp_indices = list(range(self.len_esp))
        random.shuffle(esp_indices)
        self.flip_esp_indices = set(esp_indices[:num_flips_esp])

    def __len__(self) -> int:
        return self.len_normal + self.len_esp

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx < self.len_normal:
            id = idx
            data = self.dataset_normal[id]
            label = 0
            if id in self.flip_normal_indices:
                label = 1  # contaminated
        else:
            id = idx - self.len_normal
            data = self.dataset_esp[id]
            label = 1
            if id in self.flip_esp_indices:
                label = 0  # contaminated

        output = {
            "data": data,
            "label": label,
            "id": id,
        }
        return output
