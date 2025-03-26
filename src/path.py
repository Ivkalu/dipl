import kagglehub
from pathlib import Path
import os
import pandas as pd
import json


KAGGLE_DATASET_PATH = kagglehub.dataset_download('soumendraprasad/musical-instruments-sound-dataset')
BASE_PATH = Path.cwd().parent.resolve() / "data"
CONFIG_PATH = Path.cwd() / "config.json"

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config = json.load(file)
effect_name = config.get("effect_name", "default_value")

csv_path = os.path.join(KAGGLE_DATASET_PATH, "Metadata_Train.csv")
df = pd.read_csv(csv_path)
guitar_files_only = df[df["Class"] == "Sound_Guitar"]["FileName"]
train_input_folder = os.path.join(KAGGLE_DATASET_PATH, "Train_submission", "Train_submission")

csv_path = os.path.join(KAGGLE_DATASET_PATH, "Metadata_Test.csv")
df = pd.read_csv(csv_path)
guitar_files_only_test = df[df["Class"] == "Sound_Guiatr"]["FileName"]
test_input_folder = os.path.join(KAGGLE_DATASET_PATH, "Test_submission", "Test_submission")

train_output_folder = Path(os.path.join(BASE_PATH, "train_y", effect_name))
train_output_folder.mkdir(parents=True, exist_ok=True)

test_output_folder = Path(os.path.join(BASE_PATH, "test_y", effect_name))
test_output_folder.mkdir(parents=True, exist_ok=True)

train_input_files = [os.path.join(train_input_folder, file) for file in guitar_files_only]
train_output_files = [os.path.join(train_output_folder, file) for file in guitar_files_only]

model_output_folder = Path(os.path.join(BASE_PATH, "model_output"))
model_output_folder.mkdir(parents=True, exist_ok=True)