from threading import local
import os
import math
import argparse
import numpy as np
from scipy.io import wavfile
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import IPython.display as ipd
from IPython.display import display, Audio
from bisect import bisect_left



# Pre-emphasis filter funkcija
def pre_emphasis_filter(x, coeff=0.95):
    # x je tensor shape (batch_size, seq_len)
    # primeni y[t] = x[t] - coeff * x[t-1]
    y = torch.zeros_like(x)
    y[:, 0] = x[:, 0]
    y[:, 1:] = x[:, 1:] - coeff * x[:, :-1]
    return y

# Custom loss: Error to signal ratio sa pre-emphasis filterom
def error_to_signal(y_true, y_pred):
    #y_true = pre_emphasis_filter(y_true)
    #y_true = pre_emphasis_filter(y_pred)
    numerator = torch.sum((y_true - y_pred)**2, dim=1)
    denominator = torch.sum(y_true**2, dim=1) + 1e-10
    loss = numerator / denominator
    return torch.mean(loss)


def mean_squared_error(y_true, y_pred):
    loss = torch.mean((y_true - y_pred) ** 2)
    return loss

def mean_absolute_error(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred))


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


# Model klasa
class ConvLSTMModel(nn.Module):
    def __init__(self, conv1d_filters, hidden_units):
        super(ConvLSTMModel, self).__init__()

        # Conv1D layers (PyTorch uses Conv1d with input shape: [batch, channels, sequence_length])
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=conv1d_filters,
                               kernel_size=12, padding='same')
        self.conv2 = nn.Conv1d(in_channels=conv1d_filters, out_channels=conv1d_filters,
                               kernel_size=12, padding='same')

        self.lstm = nn.LSTM(input_size=conv1d_filters, hidden_size=hidden_units, batch_first=True)
        self.fc = nn.Linear(hidden_units, 2)

    def forward(self, x):
                                    # [batch_size, input_size, channels]
        x = x.transpose(1, 2)       # [batch_size, channels, input_size]
        x = self.conv1(x)           # [batch_size, conv1d_filters, input_size]
        x = self.conv2(x)           # [batch_size, conv1d_filters, input_size]
        x = x.transpose(1, 2)       # [batch_size, input_size, conv1d_filters]
        x, _ = self.lstm(x)         # [batch_size, input_size, hidden_units]
        x = x[:, -1, :]             # [batch_size, hidden_units]
        x = self.fc(x)              # [batch_size, channels]
        return x

    def process_audio_file(self, x, y, input_size=100, batch_size=400, device='cpu'):
        """self.eval()
        #[channels, len])

        # inp = x.T.unfold(0, size=input_size, step=1).permute(0, 2, 1).to(device)
        
        predicted = []
        with torch.no_grad():
            for i in range(0, x.shape[1]):
                if not i%100: print(i)
                pred = self(x[:,max(0,i-100):i+1].T.unsqueeze(0).to(device))
                predicted.append(pred)

        predicted_audio = torch.cat(predicted).T
        return predicted_audio
        """

        self.eval()
        with torch.no_grad():
            # Build all sliding windows
            seq_len = x.shape[1]
            num_windows = seq_len
            windows = []

            for i in range(seq_len):
                start_idx = max(0, i - input_size)
                chunk = x[:, start_idx:i+1]
                # Pad to exactly input_size+1 if needed
                if chunk.shape[1] < input_size + 1:
                    pad_width = input_size + 1 - chunk.shape[1]
                    chunk = torch.cat([
                        torch.zeros((x.shape[0], pad_width), dtype=x.dtype),
                        chunk
                    ], dim=1)
                windows.append(chunk.T.unsqueeze(0))

            # Stack into shape [num_windows, input_size+1, channels]
            windows = torch.cat(windows, dim=0)

            # Run in batches
            predicted = []
            for i in range(0, num_windows, batch_size):
                batch = windows[i:i+batch_size].to(device)
                pred = self(batch)
                predicted.append(pred.cpu())

            predicted_audio = torch.cat(predicted, dim=0).T
            return predicted_audio

        
    def save(self):
        #model_path = f'models/{name}/{name}.pt'
        #torch.save({
        #    'model_state_dict': model.state_dict(),
        #    'input_size': input_size,
        #    'conv1d_filters': conv1d_filters,
        #    'conv1d_stride': conv1d_stride,
        #    'hidden_units': hidden_units
        #}, model_path)
        pass