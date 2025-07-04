import os
from typing import Any

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
    ) -> None:
        self.root = root
        self.num_past_steps = num_past_steps
        self.num_future_steps = num_future_steps

        data = self._load_data()
        self.data = data.transpose("time", "y", "x")
        self.window_size = num_past_steps + num_future_steps
        self.T, self.H, self.W = self.data.shape

        self.num_time = self.T - self.window_size + 1
        self.num_spatial = self.H * self.W
        self.total = self.num_time * self.num_spatial

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, index: int) -> dict[str, Any]:
        time_idx = index // self.num_spatial
        spatial_idx = index % self.num_spatial
        y = spatial_idx // self.W
        x = spatial_idx % self.W

        past_steps = self.data.isel(
            x=x, y=y, time=slice(time_idx, time_idx + self.num_past_steps)
        )
        future_steps = self.data.isel(
            x=x,
            y=y,
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
            "past": past_steps_normalized,
            "future": future_steps_normalized,
            "mean": mean,
            "std": std,
        }

    def _load_data(self) -> xr.DataArray:
        pathname = os.path.join(self.root, self.data_file_name)
        if os.path.exists(pathname):
            data = xr.open_zarr(pathname)
            ndvi = data.ndvi
            ndvi = ndvi.isel(x=slice(0, 50), y=slice(0, 50))
            return ndvi.compute()
        else:
            raise FileNotFoundError
