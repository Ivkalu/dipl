import numpy as np
from scipy.io import wavfile


def save_wav(name, data, sample_rate=44100):
    """
    Save multi-channel audio to a WAV file, normalized to [-1, 1].

    Args:
        name (str): Output file name.
        data (np.ndarray): Shape (num_channels, num_samples) or (num_samples, num_channels).
        sample_rate (int): Sampling rate in Hz.
    """
    # Ensure data is float32
    data = np.asarray(data, dtype=np.float32)

    # If shape is (channels, samples), transpose to (samples, channels)
    if data.ndim == 2 and data.shape[0] < data.shape[1]:
        data = data.T

    # Normalize to [-1, 1] if needed
    max_val = np.max(np.abs(data))
    if max_val > 0:
        data = data / max_val

    wavfile.write(name, sample_rate, data)
