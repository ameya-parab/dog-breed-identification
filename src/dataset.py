import os
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

sys.path.insert(0, os.path.join(os.getcwd(), ".."))

from config import BREED, CROPPED_DIM, DATA_DIR, IMAGE_DIM

from src.utils import seed_worker, set_random_seed

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
TRANSFORMS = {
    "augmented": transforms.Compose(
        [
            transforms.RandomResizedCrop(size=IMAGE_DIM),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=CROPPED_DIM),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    ),
    "nonaugmented": transforms.Compose(
        [
            transforms.Resize(size=IMAGE_DIM),
            transforms.CenterCrop(size=CROPPED_DIM),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    ),
}


class Dogs(Dataset):
    def __init__(self, split: str, dataset: pd.DataFrame, augment: bool = False):

        self.split = split
        self.dataset = dataset
        self.transform_pipeline = TRANSFORMS.get(
            "augmented" if augment else "nonaugmented"
        )

    def __len__(self):
        return len(self.dataset.index)

    def __getitem__(self, idx):

        image = self.transform_pipeline(
            Image.open(
                os.path.join(DATA_DIR, self.split, f"{self.dataset.id[idx]}.jpg")
            )
        )
        if self.split == "train":
            label = torch.tensor(BREED.index(self.dataset.breed[idx]))
        else:
            label = torch.tensor(0)

        return image, label


def fetch_dataset(random_seed: int, batch_size: int, num_workers: int = 4):

    set_random_seed(random_seed)

    train = pd.read_csv(os.path.join(DATA_DIR, "labels.csv"))
    train_idx, valid_idx = train_test_split(
        np.arange(len(train.index)),
        test_size=0.2,
        shuffle=True,
        stratify=train.breed.values,
        random_state=random_seed,
    )

    train_dataset = Dogs(
        split="train",
        dataset=train.iloc[train_idx].reset_index(drop=True),
        augment=True,
    )
    validation_dataset = Dogs(
        split="train", dataset=train.iloc[valid_idx].reset_index(drop=True)
    )

    generator = torch.Generator()
    generator.manual_seed(0)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        shuffle=True,
    )
    valid_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        shuffle=False,
    )

    test = pd.DataFrame(
        [image.split(".")[0] for image in os.listdir(os.path.join(DATA_DIR, "test"))],
        columns=["id"],
    )

    test_dataset = Dogs(split="test", dataset=test)

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        shuffle=False,
    )

    return (train_dataloader, valid_dataloader, test_dataloader)
