
import torch
from pathlib import Path
import os
import pandas as pd
from datasets.wav_dataset import WavDataset, normalize
from pedalboard import Pedalboard
DATASET_PATH="E:\\source\\dipl\\data"
from torch.utils.data import Dataset
from typing import Literal



class DataModule():
    def __init__(self, 
                 effect_name: str, 
                 input_size: int, 
                 max_samples: int, 
                 max_wav_files: int,
                 guitar_only: bool = True, 
                 batch_size: int  = 16, 
                 num_workers: int = 4, 
                 ):
        

        self.input_size = input_size
        self.max_samples = max_samples
        self.max_wav_files = max_wav_files
        self.effect_name = effect_name
        self.guitar_only = guitar_only
        self.batch_size = batch_size
        self.num_workers = num_workers

    def dataset(self, type: Literal["train", "test", "valid"]):
        t = "guitar" if self.guitar_only else "other"
        x_dir = Path(os.path.join(DATASET_PATH, type, "x", t))
        y_dir = Path(os.path.join(DATASET_PATH, type, "y", t, self.effect_name))

        return WavDataset(
            input_dir=x_dir,
            target_dir=y_dir,
            input_size=self.input_size,
            max_samples=self.max_samples, 
            max_wav_files=self.max_wav_files
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset("train"),
            batch_size=self.batch_size,
            #shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            #pin_memory=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset("test"),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            #pin_memory=True,
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset("valid"),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            #pin_memory=True,
        )
