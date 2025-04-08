import io
import pathlib

from stable_baselines3.common.base_class import BaseAlgorithm
import torch

from .item_farm_a2c import construct_ItemFarmA2C, ItemFarmA2C
from .item_farm_ppo import construct_ItemFarmPPO, ItemFarmPPO


def construct_ItemFarmPolicy(
    algorithm: str = "a2c", batch_size: int = 64, n_epochs: int = 10, *args, **kwargs
) -> BaseAlgorithm:
    assert algorithm in ["a2c", "ppo"]
    if algorithm == "a2c":
        return construct_ItemFarmA2C(*args, **kwargs)
    elif algorithm == "ppo":
        return construct_ItemFarmPPO(
            batch_size=batch_size, n_epochs=n_epochs, *args, **kwargs
        )


def load_ItemFarmPolicy(
    algorithm: str = "ppo",
    path: str | pathlib.Path | io.BufferedIOBase = "",
    device: torch.device | str = "auto",
):
    assert algorithm in ["a2c", "ppo"]
    if algorithm == "a2c":
        return ItemFarmA2C.load(path, device=device)
    elif algorithm == "ppo":
        return ItemFarmPPO.load(path, device=device)
