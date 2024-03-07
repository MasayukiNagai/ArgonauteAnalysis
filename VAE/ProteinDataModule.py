import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
import pytorch_lightning as pl


class ProteinDataModule(pl.LightningDataModule):
    def __init__(self, full_dataset, batch_size=128, weights=None, indices=None, 
                 seed_value=42):
        super().__init__()
        self.full_dataset = full_dataset
        if indices is None:
            self.dataset = self.full_dataset
        else:
            self.dataset = self.full_dataset[indices]
        self.batch_size = batch_size
        self.weights = weights
        self.num_workers = 4  ### Let Ray Tune handle this?
        self.seed = torch.Generator().manual_seed(seed_value)

    def setup(self, stage=None):
        total_size = len(self.dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        # test_size = total_size - train_size - val_size

        indices = torch.randperm(total_size, generator=self.seed)
        self.train_indices = indices[:train_size]
        self.val_indices = indices[train_size : train_size + val_size]
        self.test_indices = indices[train_size + val_size :]
        self.train_dataset = Subset(self.dataset, self.train_indices)
        self.val_dataset = Subset(self.dataset, self.val_indices)
        self.test_dataset = Subset(self.dataset, self.test_indices)

        # Prepare weights for weighted random sampling in training loader
        if self.weights is not None:
            train_weights = self.weights[self.train_dataset.indices]
            self.train_sampler = WeightedRandomSampler(
                train_weights, num_samples=len(train_weights), replacement=True
            )
        else:
            self.train_sampler = None

    def train_dataloader(self):
        if self.train_sampler is not None:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                sampler=self.train_sampler,
                shuffle=False,
                num_workers=self.num_workers,
            )
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader_indices(self):
        return self._with_indices(self.val_dataloader(), self.val_indices)

    def _with_indices(self, my_dataloader, indices):
        for i, batch in enumerate(my_dataloader):
            batch_size = len(batch)
            yield batch, indices[i * batch_size : (i + 1) * batch_size]

    def get_dataloader(self, indices):
        return DataLoader(
            Subset(self.full_dataset, indices), batch_size=self.batch_size
        )
    
    def get_train_indices(self):
        return self.train_indices

    def get_val_indices(self):
        return self.val_indices

    def get_test_indices(self):
        return self.test_indices


