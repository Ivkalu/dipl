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
from paths import *

from models.lstm import GuitarPedalLSTM
from models.wavenet import WaveNetModel
from datasets.distortion import DistortionDataModule
from models.tcn import TCN

def main():

    print_batch = True

    data_module = DistortionDataModule(
        chunk_size=1000, 
        batch_size=4
        )
    
    lstm = GuitarPedalLSTM()

    wavenet = WaveNetModel()
    
    baselineFCNet = BaselineFCNet(
        hidden_size=5, 
        hidden_layers=2
    )

    tcn = TCN()
    
    train_model(
        model=lstm, 
        data_module=data_module,
        epochs=5, 
        lr=0.001, 
        print_batch=print_batch, 
    )

if __name__ == "__main__":
    main()