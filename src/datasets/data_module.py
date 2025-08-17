
import torch
from pathlib import Path
import os
import pandas as pd
from datasets.wav_dataset import FrequencyQueueDataset, WaveformFileDataset
from pedalboard import Pedalboard
DATASET_PATH="E:\\source\\dipl\\data"
from torch.utils.data import Dataset
from typing import Literal



class DataModule():
    def __init__(self, 
                 effect_name: str, 
                 input_size: int, 
                 max_wav_files: int,
                 batch_size: int, 
                 num_workers: int = 4, 
                 guitar_only: bool = True, 
                 ):
        

        self.input_size = input_size
        self.max_wav_files = max_wav_files
        self.effect_name = effect_name
        self.guitar_only = guitar_only
        self.batch_size = batch_size
        self.num_workers = num_workers

    def dataset(self, type: Literal["train", "test", "valid"], max_samples: int):
        t = "guitar" if self.guitar_only else "other"
        x_dir = Path(os.path.join(DATASET_PATH, type, "x", t))
        y_dir = Path(os.path.join(DATASET_PATH, type, "y", t, self.effect_name))

        return FrequencyQueueDataset(
            input_dir=x_dir,
            target_dir=y_dir,
            input_size=self.input_size,
            max_samples=max_samples, 
            max_wav_files=self.max_wav_files
        )
    
    def get_waveform_dataset(self, type: Literal["train", "test", "valid"]):
        t = "guitar" if self.guitar_only else "other"
        x_dir = Path(os.path.join(DATASET_PATH, type, "x", t))
        y_dir = Path(os.path.join(DATASET_PATH, type, "y", t, self.effect_name))

        return WaveformFileDataset(
            input_dir=x_dir,
            target_dir=y_dir,
        )

    def train_dataloader(self, max_samples: int, shuffle=True):
        return torch.utils.data.DataLoader(
            self.dataset("train", max_samples=max_samples),
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=True,
            #pin_memory=True,
        )

    def test_dataloader(self, max_samples: int):
        return torch.utils.data.DataLoader(
            self.dataset("test", max_samples=max_samples),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            #pin_memory=True,
        )
    

