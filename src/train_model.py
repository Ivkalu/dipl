import torch
import torch.nn as nn
import torch.optim as optim
import os
from datasets.data_module import DataModule
from losses.mel import mel_loss

def train_model(
        model, 
        data_module: DataModule, 
        epochs=5, 
        lr=0.001, 
        print_batch=False, 
    ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()



    model.train()

    data_loader = data_module.train_dataloader()
    data_loader_test = data_module.test_dataloader()

    for epoch in range(epochs):
        total_loss = 0
        batch_num = 0
        for batch in data_loader:
            input_tensor, target_tensor = batch

            input_tensor = input_tensor.to(device)
            target_tensor = target_tensor.to(device)

            outputs = model(input_tensor)
            loss = criterion(outputs, target_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if print_batch:
                num_batches = len(data_loader)
                print(f"Batch [{batch_num+1}/{num_batches}], Loss: {loss.item():.6f}")
            batch_num += 1

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {total_loss/len(data_loader):.6f}")
        torch.save(model.state_dict(), os.path.join("checkpoints", f"{model.name}_{data_module.effect_name}_{epoch+1}.pth"))

        # ---------- Test Loss Evaluation ----------
        model.eval()

        test_loss = 0.0
        with torch.no_grad():
            for input_tensor, target_tensor in data_loader_test:
                input_tensor = input_tensor.to(device)
                target_tensor = target_tensor.to(device)

                outputs = model(input_tensor)
                loss = criterion(outputs, target_tensor)
                test_loss += loss.item()

        avg_test_loss = test_loss / len(data_loader_test)
        print(f"Epoch [{epoch+1}/{epochs}], Test Loss: {avg_test_loss:.6f}")
        model.train()