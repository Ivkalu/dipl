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
from models.tcn import TCN
from models.wavenet import WaveNetModel

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
            
            if not i%100:
                print(f"Epoch: {epoch}, Batch {i}/{len(data_loader)}, Loss: {loss.item()}")


def evaluate(data_loader, model, loss_func):
    total_loss = 0.0
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(data_loader):
            batch_x = batch_x.to(device)  # [batch_size, input_size, channels]
            batch_y = batch_y.to(device)  # [batch_size, channels]

            outputs = model(batch_x)      # [batch_size, channels]
            loss = loss_func(batch_y.unsqueeze(1), outputs.unsqueeze(1))
            total_loss += loss.item() * batch_x.size(0)  # sum over batch
            if not i%100:
                print(f"Batch {i}/{len(data_loader)}, Loss: {loss.item()}")

    avg_loss = total_loss / len(data_loader.dataset)
    return avg_loss


def main():
    """
    
    dragonflyRoomReverb,
    dragonflyPlateReverb,
    ragingDemon
    simpleDist

    """
    name = "model"
    os.makedirs('models/' + name, exist_ok=True)

    epochs = 1

    learning_rate = 0.001

    input_size = 100

    data_module = DataModule(
        effect_name="simpleDist", 
        input_size=input_size, 
        max_wav_files=11,
        batch_size=400, 
        num_workers=0)

    train_dataloader = data_module.train_dataloader(max_samples=1000*1) 
    test_dataloader = data_module.test_dataloader(max_samples=44100*1)
    test_dataset = data_module.get_waveform_dataset("test")

    model1 = BaselineFCNet().to(device)
    model2 = ConvLSTMModel().to(device)
    model3 = WaveNetModel().to(device)
    model4 = TCN().to(device)
    model5 = ConvMambaModel().to(device)
    model = model1
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    train(
        epochs=epochs, 
        data_loader=train_dataloader, 
        optimizer=optimizer,
        model=model, 
        loss_func=mean_squared_error)
    
    del train_dataloader  # free memory
    torch.cuda.empty_cache()
    
    model.eval()
    with torch.no_grad():
        
        in_data, out_data = next(iter(test_dataset))
        
        pred = model.process_whole_audio_file(in_data, out_data, input_size=input_size, device=device)
        #pred = model(in_data)

        # Save wav
        save_wav("predicted.wav", pred.cpu().numpy())
        save_wav("input.wav", in_data.cpu().numpy())
        save_wav("target.wav", out_data.cpu().numpy())
        
        esr = evaluate(
            data_loader=test_dataloader, 
            model=model, 
            loss_func=error_to_signal)
        print(f"Average ESR loss on a test dataset: {esr}")

        mse = evaluate(
            data_loader=test_dataloader, 
            model=model, 
            loss_func=mean_squared_error)
        print(f"Average MSE loss on a test dataset: {mse}")
        

    del test_dataloader  # free memory
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()