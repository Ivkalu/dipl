import torch
import torch.nn as nn
import torch.optim as optim
import os

def train_model(model, dataloader, epochs=5, lr=0.001, print_batch=False, name="model"):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()

    for epoch in range(epochs):
        total_loss = 0
        batch = 0
        for inputs, targets in dataloader:
            optimizer.zero_grad()

            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, targets)  # Compute loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if print_batch:
              print(f"Batch [{batch+1}], Loss: {loss.item():.6f}")
            batch += 1

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.6f}")
        torch.save(model.state_dict(), os.path.join("../models", f"{name}_{epoch+1}.pth"))