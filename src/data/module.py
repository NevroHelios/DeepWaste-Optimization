from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import torch
from torch import nn
import pytorch_lightning as pl
from torchvision.datasets import ImageFolder
from typing import Optional, Literal

from src.data.transforms import Transform
from pathlib import Path



class DataLoaderWrapper(pl.LightningDataModule):
    def __init__(self, 
                 data_dir:Path,
                 img_size:int=128,
                 mode:Literal['augment'] | None=None,
                 batch_size:int=32,
                 num_workers:int=4):
        super().__init__()

        transform = Transform(img_size=img_size)
        train_transformers, val_transformers = transform.get_transforms(mode=mode)

        train_dataset = ImageFolder(data_dir / 'train', transform=train_transformers)
        val_dataset = ImageFolder(data_dir / 'val', transform=val_transformers)
        test_dataset = ImageFolder(data_dir / 'test', transform=val_transformers)
        assert train_dataset.class_to_idx == test_dataset.class_to_idx, "Train and Test datasets have different class to index mapping."
        self.idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}


        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
 
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
 
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
    