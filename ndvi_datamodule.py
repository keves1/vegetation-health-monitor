from typing import Any

from torch.utils.data import DataLoader, Subset
from lightning.pytorch import LightningDataModule

from ndvi_dataset import NDVIDataset


class NDVIDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 64,
        val_split_pct: float = 0.2,
        test_split_pct: float = 0.2,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_split_pct = val_split_pct
        self.test_split_pct = test_split_pct
        self.num_workers = num_workers
        self.kwargs = kwargs

    def setup(self, stage):
        dataset = NDVIDataset(**self.kwargs)
        train_split_pct = 1 - (self.val_split_pct + self.test_split_pct)
        train_size = int(train_split_pct * len(dataset))
        val_size = int(self.val_split_pct * len(dataset))
        train_indices = range(train_size)
        val_indices = range(train_size, train_size + val_size)
        test_indices = range(train_size + val_size, len(dataset))
        self.train_dataset = Subset(dataset, train_indices)
        self.val_dataset = Subset(dataset, val_indices)
        self.test_dataset = Subset(dataset, test_indices)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
