import numpy as np
from pedalboard import Pedalboard, load_plugin, Distortion
import soundfile as sf
import os
import sys
import contextlib

@contextlib.contextmanager
def suppress_stderr():
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    old_stderr_fd = os.dup(2)  # Duplicate stderr (fd 2)

    try:
        os.dup2(devnull_fd, 2)  # Redirect fd 2 (stderr) to /dev/null
        yield
    finally:
        os.dup2(old_stderr_fd, 2)  # Restore original stderr
        os.close(old_stderr_fd)
        os.close(devnull_fd)

def get_impulse_response(board):
    impulse_duration_seconds = 50
    impulse = np.zeros((44100 * impulse_duration_seconds , 2), dtype=np.float32)
    impulse[0] = [1.0, 1.0]  # 1-sample stereo impulse

    processed_impulse = board(impulse, sample_rate=44100)
    threshold = 1e-5
    indices = np.where(np.abs(processed_impulse).max(axis=1) > threshold)[0]

    ir_length = indices[-1]

    return ir_length

def process_audio_with_pedalboard(audio, sr, board, tail_samples=0):
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=1)

    # Append silence for tail decay
    silence = np.zeros((tail_samples, audio.shape[1]), dtype=audio.dtype)
    padded_audio = np.concatenate((audio, silence), axis=0)

    processed = board(padded_audio, sample_rate=sr)

    return processed


def generate_dataset(input_dir, output_base_dir, plugin_dict, sample_rate=44100):
    if not os.path.exists(input_dir):
        print(f"❌ Input directory does not exist: {input_dir}")
        return

    audio_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".wav")]

    if not audio_files:
        print(f"⚠️ No WAV files found in {input_dir}")
        return

    for pedal_name, board in plugin_dict.items():
        print(f"\n🚀 Generating dataset for: {pedal_name}")
        output_dir = os.path.join(output_base_dir, pedal_name)
        os.makedirs(output_dir, exist_ok=True)

        for plugin in board:
            if hasattr(plugin, 'parameters'):
                keys = plugin.parameters.keys()
                print(f"plugin parameters: {keys}")

        ir_length_samples = get_impulse_response(board)
        ir_length_seconds = ir_length_samples / 44100
        print(f"⌚ Impulse response time: {ir_length_seconds:.3f} seconds, {ir_length_samples} samples\n")

        for file_name in audio_files:
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name)

            try:
                audio, sr = sf.read(input_path)
                processed_audio = process_audio_with_pedalboard(audio, sr, board, tail_samples=ir_length_samples)
                sf.write(output_path, processed_audio, sr)
            except Exception as e:
                print(f"❌ Failed to process {file_name} with {pedal_name}: {e}")





if __name__ == "__main__":
    with suppress_stderr():
        dataset_plugins = {
            "simpleDist": Pedalboard([Distortion()]),
            "ragingDemon": Pedalboard([
                Distortion(drive_db=4),
                load_plugin("E:\\source\\dipl\\plugins\\ragingdemon.vst3", parameter_values={'drive': 0.13, 'lpc': 2000})
            ]),
            "dragonflyPlateReverb": Pedalboard([
                load_plugin("E:\\source\\dipl\\plugins\\DragonflyPlateReverb.vst3", parameter_values={'decay_s': 3})
            ]),
            "dragonflyRoomReverb": Pedalboard([
                load_plugin("E:\\source\\dipl\\plugins\\DragonflyRoomReverb.vst3", parameter_values={'decay_s': 10})
            ])
            # "dragonflyEarlyReflections": Pedalboard([
            #     load_plugin("E:\\source\\dipl\\plugins\\DragonflyEarlyReflections.vst3", parameter_values={})
            # ]),
            # "dragonflyHallReverb": Pedalboard([
            #     load_plugin("E:\\source\\dipl\\plugins\\DragonflyHallReverb.vst3", parameter_values={'decay_s': 10})
            # ]),
        }

        input_dir = "E:\\source\\dipl\\data\\train\\x\\guitar"
        output_dir = "E:\\source\\dipl\\data\\train\\y\\guitar"
        generate_dataset(input_dir, output_dir, dataset_plugins)