from wav_dataset import WavDataset
import torch
from paths import KAGGLE_DATASET_PATH
from pathlib import Path
import os
import pandas as pd
from data_module import DataModule
from create_dataset import apply_effect
from pedalboard import Pedalboard, Chorus, Reverb, Distortion


class ReverbDataModule(DataModule):

    name="reverb"

    def __init__(self, **kwargs):
        
        # check if dataset exists
        
        # if not, create it

        # otherwise, init
        super().__init__(**kwargs)

    def create(self):
        # TODO make this created with different pedal, not from spotify
        board = Pedalboard([Reverb()])
        self.create_dataset_with_effect("reverb", board)
