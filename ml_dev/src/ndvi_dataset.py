import os
from typing import Any

import torch
import xarray as xr
import xbatcher as xb
from torch.utils.data import Dataset


class NDVIDataset(Dataset):
    def __init__(
        self,
        root: str,
        num_past_steps: int = 3,
        num_future_steps: int = 1,
        time_overlap: int = 10,
        batch_size: int = 100,
    ) -> None:
        self.root = root
        self.num_past_steps = num_past_steps
        self.num_future_steps = num_future_steps
        self.time_overlap = time_overlap
        self.batch_size = batch_size

        self.window_size = num_past_steps + num_future_steps
        self.bgen = self._load_data()
        # self.T, self.Z = self.data.shape

        # self.num_time = self.T - self.window_size + 1
        # self.total = self.num_time * self.Z

    def __len__(self) -> int:
        return len(self.bgen)

    def __getitem__(self, index: int) -> dict[str, Any]:
        batch = self.bgen[index].load()

        past_steps = batch.isel(time=slice(0, self.num_past_steps))
        future_steps = batch.isel(
            time=slice(self.num_past_steps, self.window_size),
        )

        past_steps_tensor: torch.Tensor = torch.tensor(
            past_steps.values, dtype=torch.float32
        )
        past_steps_tensor = past_steps_tensor.permute(1, 0).unsqueeze(
            -1
        )  # dimension order: batch, time, input size
        future_steps_tensor: torch.Tensor = torch.tensor(
            future_steps.values, dtype=torch.float32
        )
        future_steps_tensor = future_steps_tensor.permute(1, 0).unsqueeze(-1)

        mean_over_time = past_steps_tensor.mean(dim=1, keepdim=True)
        std_over_time = past_steps_tensor.std(dim=1, keepdim=True)
        past_steps_normalized = (past_steps_tensor - mean_over_time) / (
            std_over_time + 1e-12
        )
        future_steps_normalized = (future_steps_tensor - mean_over_time) / (
            std_over_time + 1e-12
        )

        return {
            "past_targets": past_steps_normalized,
            "future_targets": future_steps_normalized,
            "mean": mean_over_time,
            "std": std_over_time,
        }

    def _load_data(self) -> xb.BatchGenerator:
        if os.path.exists(f"{self.root}/zarr.json"):
            ds = xr.open_zarr(self.root)
            bgen = xb.BatchGenerator(
                ds["ndvi"],
                input_dims={"z": self.batch_size, "time": self.window_size},
                input_overlap={"time": self.time_overlap},
            )
            return bgen
        else:
            raise FileNotFoundError
