from typing import Any

from pathlib import Path

import torch
from albumentations import Compose
from dataclasses import dataclass
from torch.utils.data import DataLoader, Dataset


@dataclass
class Loader:
    train_loader: Any
    val_loader: Any
    test_loader: Any


class BaseLoader(Dataset):
    def __init__(
        self,
        root_folder: str,
        image_normalization: str,
        ground_truth_normalization: str,
        augmenters: Compose,
        mode: str,
    ):
        self.mode = mode
        self.root_folder = Path(root_folder)

        self.image_normalization = image_normalization
        self.ground_truth_normalization = ground_truth_normalization
        self.augmenters = augmenters

        self.images = sorted(list((self.root_folder / self.mode / "images").glob("*")))
        self.labels = sorted(list((self.root_folder / self.mode / "labels").glob("*")))

    @classmethod
    def get_data_loader(
        cls,
        root_folder: str,
        image_normalization: str,
        label_normalization: str,
        augmenters: Compose,
        batch_size: int,
    ):
        # Disable pin_memory for large images (batch_size <= 2) to reduce memory usage
        use_pin_memory = torch.cuda.is_available() and batch_size > 2
        
        train_data = DataLoader(
            dataset=cls(
                root_folder,
                image_normalization,
                label_normalization,
                augmenters,
                "train",
            ),
            shuffle=True,
            num_workers=0,
            batch_size=batch_size,
            pin_memory=use_pin_memory,
        )
        val_data = DataLoader(
            dataset=cls(
                root_folder, image_normalization, label_normalization, Compose([]), "val"
            ),
            shuffle=True,
            num_workers=0,
            batch_size=batch_size,
            pin_memory=use_pin_memory,
        )

        test_data = DataLoader(
            dataset=cls(
                root_folder, image_normalization, label_normalization, Compose([]), "test"
            ),
            shuffle=True,
            num_workers=0,
            batch_size=batch_size,
            pin_memory=use_pin_memory,
        )
        return Loader(train_data, val_data, test_data)

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
