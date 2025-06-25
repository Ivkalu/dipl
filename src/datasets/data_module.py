
import torch
from paths import KAGGLE_DATASET_PATH, BASE_PATH
from pathlib import Path
import os
import pandas as pd
from datasets.wav_dataset import WavDataset
from datasets.chunk_dataset import ChunkedWavDataset
from pedalboard import Pedalboard
from datasets.create_dataset import apply_effect


class DataModule():
    train_input_folder = os.path.join(KAGGLE_DATASET_PATH, "Train_submission", "Train_submission")
    test_input_folder = os.path.join(KAGGLE_DATASET_PATH, "Test_submission", "Test_submission")

    csv_path = os.path.join(KAGGLE_DATASET_PATH, "Metadata_Train.csv")
    df = pd.read_csv(csv_path)
    guitar_files_only_train = df[df["Class"] == "Sound_Guitar"]["FileName"]

    csv_path = os.path.join(KAGGLE_DATASET_PATH, "Metadata_Test.csv")
    df = pd.read_csv(csv_path)
    guitar_files_only_test = df[df["Class"] == "Sound_Guiatr"]["FileName"]

    # this folder is for trying out model on real files, same for all models, pedals
    model_output_folder = Path(os.path.join(BASE_PATH, "model_output"))
    model_output_folder.mkdir(parents=True, exist_ok=True)


    def __init__(self, batch_size=4, num_workers=0, chunk_size=110250):
        # different for all pedals
        self.train_output_folder = Path(os.path.join(BASE_PATH, "train_y", self.name))
        self.test_output_folder = Path(os.path.join(BASE_PATH, "test_y", self.name))
        
        # create only if missing (if)
        self.train_output_folder.mkdir(parents=True, exist_ok=True)

        # create only if missing (if)
        self.test_output_folder.mkdir(parents=True, exist_ok=True)

        # standardised list that will actually be used for dataloaders
        self.train_input_files = [os.path.join(self.train_input_folder, file) for file in self.guitar_files_only_train]
        self.train_output_files = [os.path.join(self.train_output_folder, file) for file in self.guitar_files_only_train]

        #self.train_dataset = WavDataset(
        #    input_wav_paths=self.train_input_files,
        #    output_wav_paths=self.train_output_files
        #)

        self.train_dataset = ChunkedWavDataset(
            input_wav_paths=self.train_input_files,
            output_wav_paths=self.train_output_files,
            chunk_size=chunk_size
        )

        self.batch_size=batch_size
        self.num_workers=num_workers
        self._train_dataloader = None

    @property
    def train_dataloader(self):
        if self._train_dataloader is None:
            self._train_dataloader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=True,
                pin_memory=True,
            )
        return self._train_dataloader

    def val_dataloader(self):
        raise NotImplementedError()
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        raise NotImplementedError()
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def create_dataset_with_effect(self, board: Pedalboard) -> None:
        apply_effect(self.train_input_folder, self.train_output_folder, self.guitar_files_only_train, board)
        apply_effect(self.test_input_folder, self.test_output_folder, self.guitar_files_only_test, board)
        

