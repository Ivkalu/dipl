import kagglehub
from pathlib import Path
import os
import pandas as pd
import json


KAGGLE_DATASET_PATH = kagglehub.dataset_download('soumendraprasad/musical-instruments-sound-dataset')

BASE_PATH = Path(__file__).parent.resolve().parent / "data"
#CONFIG_PATH = Path(__file__).parent.resolve() / "config.json"

#with open(CONFIG_PATH, "r", encoding="utf-8") as file:
#    config = json.load(file)
#effect_name = config.get("effect_name", "default_value")
