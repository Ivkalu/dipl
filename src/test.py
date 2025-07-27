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


# Normalize funkcija
def normalize(data):
    data_max = np.max(data)
    data_min = np.min(data)
    data_norm = max(abs(data_max), abs(data_min))
    return data / data_norm

class GuitarDataset(Dataset):
    
    """
    def __init__(self, input_dir, target_dir, input_size):
        self.input_size = input_size
        self.pairs = []  # List of (input_file_path, target_file_path)
        self.file_lengths = []  # Number of valid samples per file
        self.cumulative_offsets = []  # For fast index mapping

        input_files = sorted(os.listdir(input_dir))
        offset = 0
        for fname in input_files:
            in_path = os.path.join(input_dir, fname)
            out_path = os.path.join(target_dir, fname)
            if not os.path.exists(out_path):
                continue

            in_rate, in_data = wavfile.read(in_path)
            out_rate, out_data = wavfile.read(out_path)
            if in_rate != 44100 or out_rate != 44100:
                continue

            min_len = min(len(in_data), len(out_data))
            num_windows = min_len - input_size

            if num_windows <= 0:
                continue

            self.pairs.append((in_path, out_path))
            self.file_lengths.append(num_windows)
            self.cumulative_offsets.append(offset)
            offset += num_windows

        self.total_windows = offset
    """

    def __init__(self, input_dir, target_dir, input_size, max_samples=1000):
        self.input_size = input_size
        self.pairs = []
        self.file_lengths = []
        self.cumulative_offsets = []

        input_files = sorted(os.listdir(input_dir))
        offset = 0
        for fname in input_files:
            in_path = os.path.join(input_dir, fname)
            out_path = os.path.join(target_dir, fname)
            if not os.path.exists(out_path):
                continue

            in_rate, in_data = wavfile.read(in_path)
            out_rate, out_data = wavfile.read(out_path)
            if in_rate != 44100 or out_rate != 44100:
                continue

            min_len = min(len(in_data), len(out_data))
            num_windows = min_len - input_size

            if num_windows <= 0:
                continue

            # If adding this file would exceed max_samples, cut it
            if max_samples is not None and offset + num_windows > max_samples:
                num_windows = max_samples - offset
                if num_windows <= 0:
                    break

            self.pairs.append((in_path, out_path))
            self.file_lengths.append(num_windows)
            self.cumulative_offsets.append(offset)
            offset += num_windows

            if max_samples is not None and offset >= max_samples:
                break

        self.total_windows = offset



    def __len__(self):
        return self.total_windows

    def __getitem__(self, idx):
        # Find which file the idx corresponds to
        file_idx = None
        for i in range(len(self.cumulative_offsets) - 1):
            if self.cumulative_offsets[i] <= idx < self.cumulative_offsets[i + 1]:
                file_idx = i
                break
        if file_idx is None:
            file_idx = len(self.cumulative_offsets) - 1

        local_idx = idx - self.cumulative_offsets[file_idx]

        in_path, out_path = self.pairs[file_idx]
        in_rate, in_data = wavfile.read(in_path)
        out_rate, out_data = wavfile.read(out_path)

        in_data = normalize(in_data.astype(np.float32)).reshape(-1)
        out_data = normalize(out_data.astype(np.float32)).reshape(-1)

        min_len = min(len(in_data), len(out_data))
        in_data = in_data[:min_len]
        out_data = out_data[:min_len]

        x_window = in_data[local_idx : local_idx + self.input_size]
        y_value = out_data[local_idx + self.input_size - 1]

        x_tensor = torch.tensor(x_window).unsqueeze(-1)  # (input_size, 1)
        y_tensor = torch.tensor(y_value)

        return x_tensor, y_tensor
    



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

def save_wav(name, data):
    wavfile.write(name, 44100, data.flatten().astype(np.float32))

# Model klasa
class GuitarAmpModel(nn.Module):
    def compute_same_padding(self, kernel_size, stride):
        pad = max((stride - 1) + kernel_size - stride, 0)
        pad_left = pad // 2
        pad_right = pad - pad_left
        return pad_left, pad_right

    def __init__(self, conv1d_filters, conv1d_stride, hidden_units, input_size):
        super().__init__()
        pad1_left, pad1_right = self.compute_same_padding(12, conv1d_stride)
        self.pad1 = nn.ConstantPad1d((pad1_left, pad1_right), 0)
        self.conv1 = nn.Conv1d(1, conv1d_filters, 12, stride=conv1d_stride)

        pad2_left, pad2_right = self.compute_same_padding(12, conv1d_stride)
        self.pad2 = nn.ConstantPad1d((pad2_left, pad2_right), 0)
        self.conv2 = nn.Conv1d(conv1d_filters, conv1d_filters, 12, stride=conv1d_stride)

        self.lstm = nn.LSTM(input_size=conv1d_filters, hidden_size=hidden_units, batch_first=True)
        self.fc = nn.Linear(hidden_units, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, 1)
        x = x.permute(0, 2, 1)  # (batch, channels=1, seq_len)
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.pad2(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 1)  # (batch, seq_len, features)
        # LSTM expects (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Uzmi poslednji vremenski korak
        last_out = lstm_out[:, -1, :]  # (batch, hidden_units)
        out = self.fc(last_out)         # (batch, 1)
        return out.squeeze(1)           # (batch,)




def main():
    name = "model"
    os.makedirs('models/' + name, exist_ok=True)

    epochs = 10

    learning_rate = 0.01
    conv1d_stride = 12
    conv1d_filters = 16
    hidden_units = 36

    input_dir = "E:/source/dipl/data/train/x/guitar/"
    target_dir = "E:/source/dipl/data/train/y/guitar/simpleDist/"
    input_size = 100

    dataset = GuitarDataset(input_dir, target_dir, input_size)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    
    # Napravi model i optimizer
    model = GuitarAmpModel(conv1d_filters, conv1d_stride, hidden_units, input_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Treniranje
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = error_to_signal(batch_y.unsqueeze(1), outputs.unsqueeze(1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.6f}")

    # Save model i input_size info
    model_path = f'models/{name}/{name}.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': input_size,
        'conv1d_filters': conv1d_filters,
        'conv1d_stride': conv1d_stride,
        'hidden_units': hidden_units
    }, model_path)


    model.eval()
    with torch.no_grad():
        input_dir = "E:/source/dipl/data/train/x/guitar/"
        target_dir = "E:/source/dipl/data/train/y/guitar/simpleDist/"

        dataset = GuitarDataset(input_dir, target_dir, input_size)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

        # Assuming just one batch for this evaluation
        batch_x, batch_y = next(iter(dataloader))
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = error_to_signal(batch_y.unsqueeze(1), outputs.unsqueeze(1))

        # Sampling rate (assumed 44100 Hz)
        sr = 44100

        # Move tensors to CPU and flatten for audio playback
        pred_audio = outputs.detach().cpu().flatten().numpy()
        input_audio = batch_x.detach().cpu().flatten().numpy()
        target_audio = batch_y.detach().cpu().flatten().numpy()

        print("Predicted audio:")
        display(Audio(pred_audio, rate=sr))

        print("Input audio (test segment):")
        display(Audio(input_audio, rate=sr))

        print("Target audio (test segment):")
        display(Audio(target_audio, rate=sr))


if __name__ == "__main__":
    main()