import torch
from paths import KAGGLE_DATASET_PATH
from pathlib import Path
import os
import pandas as pd
from datasets.data_module import DataModule
from pedalboard import Pedalboard, Chorus, Reverb, Distortion

class DistortionDataModule(DataModule):

    name="distortion"

    def __init__(self, **kwargs):
        
        # check if dataset exists
        
        # if not, create it

        # otherwise, init
        super().__init__(**kwargs)


    def create(self):
        # TODO make this created with different pedal, not from spotify
        board = Pedalboard([Distortion(drive_db=25)])
        self.create_dataset_with_effect("distortion", board)