import os

import pandas as pd
import torch

ROOT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pth")

STORAGE = f"sqlite:///{os.path.join(ROOT_DIR, 'study', 'hyperparameter_studies.db')}"

BREED = sorted(pd.read_csv(os.path.join(DATA_DIR, "labels.csv")).breed.unique())
IMAGE_DIM = 128
CROPPED_DIM = 100

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
