from typing import Any

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from .ndvi_dataset import NDVIDataset


class NDVIDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 64,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.kwargs = kwargs

    def setup(self, stage):
        self.train_dataset = NDVIDataset(
            root=self.data_dir / "train_set.zarr",
            batch_size=self.batch_size,
            **self.kwargs,
        )
        self.val_dataset = NDVIDataset(
            root=self.data_dir / "val_set.zarr",
            batch_size=self.batch_size,
            **self.kwargs,
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=None, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=None)
