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
from datasets.wav_dataset import WavDataset, normalize
from models.guitarAmp import ConvLSTMModel, error_to_signal, save_wav, mean_squared_error
from datasets.data_module import DataModule
from models.baselineFCNet import BaselineFCNet




def main():
    name = "model"
    os.makedirs('models/' + name, exist_ok=True)

    epochs = 10

    learning_rate = 0.01 
    conv1d_stride = 12    
    conv1d_filters = 16
    hidden_units = 36

    input_size = 1

    dataset = DataModule(
        effect_name="simpleDist", 
        input_size=input_size, 
        max_samples=44100*10, 
        max_wav_files=12, 
        batch_size=1400, 
        num_workers=0)

    # Napravi model i optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConvLSTMModel(conv1d_filters, conv1d_stride, hidden_units, input_size).to(device)
    #model = BaselineFCNet(input_size=2, output_size=2).to(device)

    print(torch.cuda.get_device_name(0))

    print(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loader = dataset.train_dataloader()

    # Treniranje
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for i, (batch_x, batch_y) in enumerate(loader):
            batch_x = batch_x.to(device) # [batch_size, input_size, channels]
            batch_y = batch_y.to(device) # [batch_size, channels]

            #breakpoint()
            optimizer.zero_grad()
            outputs = model(batch_x) # [batch_size, channels]
            #breakpoint()
            loss = error_to_signal(batch_y.unsqueeze(1), outputs.unsqueeze(1))
            #loss = mean_squared_error(batch_y.unsqueeze(1), outputs.unsqueeze(1))
            loss.backward()
            a = loss.item()
            epoch_loss += a
            
            print(f"Batch {i+1}/{len(loader)}, {a}")
        optimizer.step()

        optimizer.learn_rate = learning_rate * (0.1 ** (epoch // 2))  # Learning rate decay every 3 epochs
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(loader):.6f}")

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
        input_dir = "E:/source/dipl/data/test/x/guitar/"
        target_dir = "E:/source/dipl/data/test/y/guitar/simpleDist/"

        # Pick one pair of full input and target audio
        input_files = sorted(os.listdir(input_dir))
        fname = input_files[0]  # Evaluate on the first file
        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(target_dir, fname)

        # Read and normalize audio
        in_rate, in_data = wavfile.read(in_path)
        out_rate, out_data = wavfile.read(out_path)
        assert in_rate == 44100 and out_rate == 44100

        in_data = normalize(in_data.astype(np.float32)).reshape(-1)
        out_data = normalize(out_data.astype(np.float32)).reshape(-1)
        min_len = min(len(in_data), len(out_data))

        in_data = in_data[:min_len]
        out_data = out_data[:min_len]

        # Prepare model inputs with sliding window
        input_windows = []
        for i in range(min_len - input_size):
            window = in_data[i : i + input_size]
            input_windows.append(torch.tensor(window).unsqueeze(1))  # (input_size, 1)

        input_tensor = torch.stack(input_windows).to(device)  # (N, input_size, 1)

        # Run the model on the full sequence
        predicted = []
        batch_size = 64
        for i in range(0, len(input_tensor), batch_size):
            batch = input_tensor[i:i+batch_size]
            out = model(batch)
            predicted.append(out.detach().cpu())

        predicted_audio = torch.cat(predicted).numpy()

        # Match length to original for fair comparison
        input_audio = in_data[input_size - 1 : min_len - 1]
        target_audio = out_data[input_size - 1 : min_len - 1]

        print("Predicted audio (full file):")
        display(Audio(predicted_audio, rate=44100))

        print("Input audio (corresponding input):")
        display(Audio(input_audio, rate=44100))

        print("Target audio (ground truth):")
        display(Audio(target_audio, rate=44100))

        # Optional: Save to WAV for inspection
        save_wav(f"predicted_{fname}", predicted_audio)
        save_wav(f"input_{fname}", input_audio)
        save_wav(f"target_{fname}", target_audio)


if __name__ == "__main__":
    main()