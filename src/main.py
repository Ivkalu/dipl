from pedalboard import Pedalboard, Chorus, Reverb, Distortion
from pedalboard.io import AudioFile
import os
from pathlib import Path

import pandas as pd
from IPython.display import Audio

import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import Dataset, DataLoader
import IPython.display as ipd

from models.baselineFCNet import BaselineFCNet
from train_model import train_model

from models.lstm import GuitarPedalLSTM
from models.wavenet import WaveNetModel
from datasets.data_module import DataModule

from models.tcn import TCN


def main():

    print_batch = True

    data_module = DataModule(
        effect_name="simpleDist",
        batch_size=40,
        chunk_size=10000, 
        num_workers=4,
        )
    
    lstm = GuitarPedalLSTM()

    wavenet = WaveNetModel()
    
    baselineFCNet = BaselineFCNet(
        hidden_size=5, 
        hidden_layers=2
    )

    tcn = TCN()
    
    train_model(
        model=baselineFCNet, 
        data_module=data_module,
        epochs=10, 
        lr=0.001, 
        print_batch=print_batch, 
    )

if __name__ == "__main__":
    main()