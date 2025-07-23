
import torch
from pathlib import Path
import os
import pandas as pd
from datasets.wav_dataset import WavDataset
from datasets.chunk_dataset import ChunkedWavDataset
from pedalboard import Pedalboard
DATASET_PATH="E:\\source\\dipl\\data"
from torch.utils.data import Dataset
from typing import Literal

class DataModule():
    def __init__(self, 
                 effect_name: str, 
                 guitar_only: bool = True, 
                 batch_size: int  = 400, 
                 num_workers: int = 0, 
                 chunk_size: int = 110250
                 ):
        

        self.effect_name = effect_name
        self.guitar_only = guitar_only
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.chunk_size = chunk_size    

    def dataset(self, type: Literal["train", "test", "valid"]):
        t = "guitar" if self.guitar_only else "other"
        x_dir = Path(os.path.join(DATASET_PATH, type, "x", t))
        y_dir = Path(os.path.join(DATASET_PATH, type, "y", t, self.effect_name))
        
        x = sorted(list(x_dir.glob("*.wav")))
        y = sorted(list(y_dir.glob("*.wav")))

        return WavDataset(
            input_wav_paths=x,
            output_wav_paths=y,
            max_length=self.chunk_size
            #chunk_size=self.chunk_size
        )

    def train_dataloader(self):

        return torch.utils.data.DataLoader(
            self.dataset("train"),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset("test"),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset("valid"),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
