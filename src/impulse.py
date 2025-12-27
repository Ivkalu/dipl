import numpy as np
import matplotlib.pyplot as plt
from pedalboard import Pedalboard, load_plugin, Distortion


def get_impulse_response(board, sr=44100, duration=50):
    impulse = np.zeros((sr * duration, 2), dtype=np.float32)
    impulse[0] = [1.0, 1.0]
    processed = board(impulse, sample_rate=sr)
    # convert to mono
    mono = processed.mean(axis=1)
    # detect where response ends
    threshold = 1e-5
    indices = np.where(np.abs(mono) > threshold)[0]
    if len(indices) == 0:
        end = sr  # fallback 1s
    else:
        end = indices[-1] + sr // 10  # add small margin
    # remove the initial impulse sample at t=0
    return mono[1:end]


def plot_ir(ir, name, sr=44100):
    t = np.arange(len(ir)) / sr
    plt.figure(figsize=(10, 4))
    plt.plot(t, ir, label="Mono")
    plt.title(f"Impulse Response - {name}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{name}_impulse_response.png")
    plt.close()


if __name__ == "__main__":
    plugins = {
        "simpleDist": Pedalboard([Distortion()]),
        "ragingDemon": Pedalboard([
            Distortion(drive_db=4),
            load_plugin("E:\\source\\dipl\\plugins\\ragingdemon.vst3", parameter_values={"drive": 0.13, "lpc": 2000})
        ]),
        "dragonflyPlateReverb": Pedalboard([
            load_plugin("E:\\source\\dipl\\plugins\\DragonflyPlateReverb.vst3", parameter_values={"decay_s": 3})
        ]),
        "dragonflyRoomReverb": Pedalboard([
            load_plugin("E:\\source\\dipl\\plugins\\DragonflyRoomReverb.vst3", parameter_values={"decay_s": 10})
        ])
    }

    for name, board in plugins.items():
        print(f"Processing {name}...")
        ir = get_impulse_response(board)
        plot_ir(ir, name)
    print("✅ Impulse response plots saved!")