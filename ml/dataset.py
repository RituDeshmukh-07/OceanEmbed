"""
PyTorch Dataset wrapping the preprocessed, already-normalized sample
arrays produced by preprocessing.py.

Shapes
------
inputs  : [N, C, PATCH_SIZE, PATCH_SIZE]  float32
            C = len(config.INPUT_VARIABLES)  (2 in the SST+SSS prototype)
            A single __getitem__ call returns one sample of shape
            [C, PATCH_SIZE, PATCH_SIZE]; the DataLoader stacks these into
            batches of shape [B, C, PATCH_SIZE, PATCH_SIZE].
targets : [N, N_DEPTHS]  float32
            A single __getitem__ call returns [N_DEPTHS]; batched shape is
            [B, N_DEPTHS] (N_DEPTHS = 15 by default).
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ml import config
from ml.preprocessing import load_split


class OceanPatchDataset(Dataset):
    """One sample = (local surface patch) -> (subsurface temperature profile)."""

    def __init__(self, npz_path: Path):
        data = load_split(npz_path)
        # [N, C, H, W] float32, already normalized in preprocessing.py
        self.inputs = torch.from_numpy(data["inputs"]).float()
        # [N, N_DEPTHS] float32, already normalized in preprocessing.py
        self.targets = torch.from_numpy(data["targets"]).float()
        self.times = data["times"]
        self.lats = data["lats"]
        self.lons = data["lons"]

        assert self.inputs.shape[0] == self.targets.shape[0], (
            "Mismatched number of input/target samples in "
            f"{npz_path}: {self.inputs.shape[0]} vs {self.targets.shape[0]}"
        )
        assert self.inputs.shape[1] == len(config.INPUT_VARIABLES), (
            f"Expected {len(config.INPUT_VARIABLES)} input channels "
            f"({config.INPUT_VARIABLES}), got {self.inputs.shape[1]}. "
            f"Re-run preprocessing after changing config.INPUT_VARIABLES."
        )
        assert self.targets.shape[1] == config.N_DEPTHS, (
            f"Expected {config.N_DEPTHS} target depths, got {self.targets.shape[1]}."
        )

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [C, PATCH_SIZE, PATCH_SIZE], y: [N_DEPTHS]
        return self.inputs[idx], self.targets[idx]


def get_dataloader(split: str, batch_size: int = None, shuffle: bool = None) -> DataLoader:
    """split in {'train', 'val', 'test'}."""
    path_map = {
        "train": config.TRAIN_SAMPLES_PATH,
        "val": config.VAL_SAMPLES_PATH,
        "test": config.TEST_SAMPLES_PATH,
    }
    if split not in path_map:
        raise ValueError(f"split must be one of {list(path_map)}, got {split!r}")

    dataset = OceanPatchDataset(path_map[split])
    if shuffle is None:
        shuffle = (split == "train")
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.NUM_WORKERS,
        drop_last=False,
    )


if __name__ == "__main__":
    # quick smoke test
    for split in ["train", "val", "test"]:
        loader = get_dataloader(split)
        x, y = next(iter(loader))
        print(f"{split}: dataset size={len(loader.dataset)}, "
              f"batch x={tuple(x.shape)}, batch y={tuple(y.shape)}")
