import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def train(epochs, data_loader, optimizer, model, loss_func):
    p = len(data_loader) // 100 + 1
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
            
            if not i%p:
                print(f"Epoch: {epoch}, Batch {i+1}/{len(data_loader)}, Loss: {loss.item()}")


def evaluate(data_loader, model, loss_func):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(data_loader):
            batch_x = batch_x.to(device)  # [batch_size, input_size, channels]
            batch_y = batch_y.to(device)  # [batch_size, channels]

            outputs = model(batch_x)      # [batch_size, channels]
            loss = loss_func(batch_y.unsqueeze(1), outputs.unsqueeze(1))
            total_loss += loss.item() * batch_x.size(0)  # sum over batch
            #if not i%50:
            #    print(f"Batch {i}/{len(data_loader)}, Loss: {loss.item()}")

    return total_loss / len(data_loader)