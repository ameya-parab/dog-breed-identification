import os

import pandas as pd
import torch
from PIL import Image
from sklearn import preprocessing
from torch.utils.data import Dataset
from torchvision import transforms

DATA_DIR = os.path.join("..", "data")
BREED = sorted(pd.read_csv(os.path.join(DATA_DIR, "labels.csv")).breed.unique())
IMAGE_DIM = 100


class Dogs(Dataset):
    def __init__(self, split: str):

        self.split = split
        self.image_files = os.listdir(os.path.join(DATA_DIR, split))
        if split == "train":
            label_encoder = preprocessing.LabelEncoder()
            label_encoder.fit(BREED)
            self.labels = torch.from_numpy(
                label_encoder.transform(
                    pd.read_csv(os.path.join(DATA_DIR, "labels.csv")).breed.values
                )
            ).type(torch.LongTensor)
        else:
            self.labels = torch.zeros(len(self.image_files)).type(torch.LongTensor)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):

        transform_pipeline = transforms.Compose(
            [
                transforms.Resize((IMAGE_DIM, IMAGE_DIM)),
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        return (
            transform_pipeline(
                Image.open(os.path.join(DATA_DIR, self.split, self.image_files[idx]))
            ),
            self.labels[idx],
        )
