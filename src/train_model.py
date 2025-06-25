import torch
import torch.nn as nn
import torch.optim as optim
import os
from datasets.data_module import DataModule

def train_model(
        model, 
        data_module: DataModule, 
        epochs=5, 
        lr=0.001, 
        print_batch=False, 
    ):
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()

    for epoch in range(epochs):
        total_loss = 0
        batch_num = 0
        for batch in data_module.train_dataloader:
            input_tensor, target_tensor = batch  # shape: (batch, seq_len, 1)

            outputs = model(input_tensor)  # Forward pass
            loss = criterion(outputs, target_tensor)  # Compute loss
            optimizer.zero_grad()

            loss.backward(retain_graph=True)
            optimizer.step()
            

            total_loss += loss.item()
            if print_batch:
                num_batches = len(data_module.train_dataloader)
                print(f"Batch [{batch_num+1}/{num_batches}], Loss: {loss.item():.6f}")
            batch_num += 1

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.6f}")
        torch.save(model.state_dict(), os.path.join("models", f"{model.name}_{data_module.name}_{epoch+1}.pth"))