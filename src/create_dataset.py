import os
from pedalboard.io import AudioFile
from pedalboard import Pedalboard, Chorus, Reverb, Distortion
from path import *


def apply_effect(
    input_folder_path: Path, 
    output_folder_path: Path, 
    file_names: list[Path], 
    pedalboard: Pedalboard) -> None:
  
    for file_name in file_names:
        file_input_path = os.path.join(input_folder_path, file_name)
        output_file_path = os.path.join(output_folder_path, file_name)

        with AudioFile(file_input_path) as f, AudioFile(output_file_path, 'w', f.samplerate, f.num_channels) as o:
            while f.tell() < f.frames:
                chunk = f.read(f.samplerate)
                effected = pedalboard(chunk, f.samplerate, reset=False)
                o.write(effected)


def create_dataset_with_effect(board: Pedalboard) -> None:
    apply_effect(train_input_folder, train_output_folder, guitar_files_only, board)
    apply_effect(test_input_folder, test_output_folder, guitar_files_only_test, board)

def main():
    board = Pedalboard([Distortion(drive_db=25)])
    create_dataset_with_effect("distortion", board)

    board = Pedalboard([Reverb()])
    create_dataset_with_effect("reverb", board)

if __name__ == "__main__":
    main()