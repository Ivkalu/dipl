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
from models.guitarAmp import ConvLSTMModel, error_to_signal, save_wav, mean_squared_error
from datasets.data_module import DataModule
from models.baselineFCNet import BaselineFCNet
from models.mamba import ConvMambaModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(epochs, data_loader, optimizer, model, loss_func):
    for epoch in range(epochs):
        epoch_loss = 0
        for i, (batch_x, batch_y) in enumerate(data_loader):
            batch_x = batch_x.to(device) # [batch_size, input_size, channels]
            batch_y = batch_y.to(device) # [batch_size, channels]

            outputs = model(batch_x) # [batch_size, channels]
            loss = loss_func(batch_y.unsqueeze(1), outputs.unsqueeze(1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            if i%100:
                print(f"Epoch: {epoch}, Batch {i+1}/{len(data_loader)}, Loss: {loss.item()}")


def evaluate(data_loader, model, loss_func):
    total_loss = 0.0
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(data_loader):
            batch_x = batch_x.to(device)  # [batch_size, input_size, channels]
            batch_y = batch_y.to(device)  # [batch_size, channels]

            outputs = model(batch_x)      # [batch_size, channels]
            loss = loss_func(batch_y.unsqueeze(1), outputs.unsqueeze(1))
            total_loss += loss.item() * batch_x.size(0)  # sum over batch
            if i%100:
                print(f"Batch {i+1}/{len(data_loader)}, Loss: {loss.item()}")

    avg_loss = total_loss / len(data_loader.dataset)
    return avg_loss


def main():
    name = "model"
    os.makedirs('models/' + name, exist_ok=True)

    epochs = 1

    learning_rate = 0.001 
    conv1d_filters = 16
    hidden_units = 26

    input_size = 100

    data_module = DataModule(
        effect_name="dragonflyRoomReverb", 
        input_size=input_size, 
        max_wav_files=11,
        batch_size=2100, 
        num_workers=0)

    train_dataloader = data_module.train_dataloader(max_samples=44100*10) 
    test_dataloader = data_module.test_dataloader(max_samples=44100*1)
    test_dataset = data_module.get_waveform_dataset("test")

    #model = ConvLSTMModel(conv1d_filters, hidden_units).to(device)
    model = ConvMambaModel(conv1d_filters, hidden_units).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    train(
        epochs=epochs, 
        data_loader=train_dataloader, 
        optimizer=optimizer, 
        model=model, 
        loss_func=error_to_signal)
    
    del train_dataloader  # free memory
    torch.cuda.empty_cache()
    
    model.eval()
    with torch.no_grad():
    
        in_data, out_data = next(iter(test_dataset))
        pred = model.process_audio_file(in_data, out_data, input_size=input_size, device=device)

        # Save wav
        save_wav("predicted.wav", pred.cpu().numpy())
        save_wav("input.wav", in_data.cpu().numpy())
        save_wav("target.wav", out_data.cpu().numpy())

        test_loss = evaluate(
            data_loader=test_dataloader, 
            model=model, 
            loss_func=error_to_signal)
        
        print(f"Average loss on a test dataset: {test_loss}")

    del test_dataloader  # free memory
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()