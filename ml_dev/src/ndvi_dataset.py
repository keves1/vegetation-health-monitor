import os
from typing import Any

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset


class NDVIDataset(Dataset):
    data_file_name = "ndvi_processed.zarr"

    def __init__(
        self,
        root: str = "data",
        num_past_steps: int = 3,
        num_future_steps: int = 1,
        num_locations: int = 100,
    ) -> None:
        self.root = root
        self.num_past_steps = num_past_steps
        self.num_future_steps = num_future_steps
        self.num_locations = num_locations

        self.data = self._load_data()
        self.window_size = num_past_steps + num_future_steps
        self.T, self.Z = self.data.shape

        self.num_time = self.T - self.window_size + 1
        self.total = self.num_time * self.Z

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, index: int) -> dict[str, Any]:
        time_idx = index // self.Z
        z_idx = index % self.Z

        past_steps = self.data.isel(
            z=z_idx, time=slice(time_idx, time_idx + self.num_past_steps)
        )
        future_steps = self.data.isel(
            z=z_idx,
            time=slice(
                time_idx + self.num_past_steps,
                time_idx + self.window_size,
            ),
        )

        past_steps_tensor: torch.Tensor = torch.tensor(
            past_steps.values, dtype=torch.float32
        ).unsqueeze(-1)
        future_steps_tensor: torch.Tensor = torch.tensor(
            future_steps.values, dtype=torch.float32
        ).unsqueeze(-1)

        mean = past_steps_tensor.mean(dim=0, keepdim=True)
        std = past_steps_tensor.std(dim=0, keepdim=True)
        past_steps_normalized = (past_steps_tensor - mean) / (std + 1e-12)
        future_steps_normalized = (future_steps_tensor - mean) / (std + 1e-12)

        return {
            "past_targets": past_steps_normalized,
            "future_targets": future_steps_normalized,
            "mean": mean,
            "std": std,
        }

    def _load_data(self) -> xr.DataArray:
        pathname = os.path.join(self.root, self.data_file_name)
        if os.path.exists(pathname):
            data = xr.open_zarr(pathname)
            flat = data.stack(z=("x", "y"))
            rng = np.random.default_rng(seed=0)  # fixed seed for reproducibility
            sample_indices = rng.choice(
                flat.z.size, size=self.num_locations, replace=False
            )
            sampled = flat.isel(z=sample_indices)
            ndvi = sampled["ndvi_8d_processed"]
            return ndvi.compute()
        else:
            raise FileNotFoundError
