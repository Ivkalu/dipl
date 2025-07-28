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
    y_true_f = pre_emphasis_filter(y_true)
    y_pred_f = pre_emphasis_filter(y_pred)
    numerator = torch.sum((y_true_f - y_pred_f)**2, dim=1)
    denominator = torch.sum(y_true_f**2, dim=1) + 1e-10
    loss = numerator / denominator
    return torch.mean(loss)


def mean_squared_error(y_true, y_pred):
    loss = torch.mean((y_true - y_pred) ** 2)
    return loss


def save_wav(name, data):
    wavfile.write(name, 44100, data.flatten().astype(np.float32))




# Model klasa
class ConvLSTMModel(nn.Module):
    def __init__(self, input_size, conv1d_filters, conv1d_strides, hidden_units):
        super(ConvLSTMModel, self).__init__()

        # Conv1D layers (PyTorch uses Conv1d with input shape: [batch, channels, sequence_length])
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=conv1d_filters,
                               kernel_size=12, stride=conv1d_strides, padding=6)
        self.conv2 = nn.Conv1d(in_channels=conv1d_filters, out_channels=conv1d_filters,
                               kernel_size=12, stride=conv1d_strides, padding=6)

        # You'll need to compute the output sequence length after the conv layers to set up LSTM input
        conv_output_length = input_size
        for _ in range(2):  # two conv layers
            conv_output_length = (conv_output_length + 2*6 - 12) // conv1d_strides + 1

        self.lstm = nn.LSTM(input_size=conv1d_filters, hidden_size=hidden_units, batch_first=True)
        self.fc = nn.Linear(hidden_units, 2)

    def forward(self, x):
        # x: [batch, seq_len, 1] => transpose for Conv1d: [batch, 1, seq_len]
        x = x.transpose(1, 2)

        x = self.conv1(x)  # [batch, filters, seq_len']
        x = self.conv2(x)

        # transpose back for LSTM: [batch, seq_len, features]
        x = x.transpose(1, 2)

        x, _ = self.lstm(x)  # [batch, seq_len, hidden]
        x = x[:, -1, :]  # take last time step

        x = self.fc(x)
        return x

