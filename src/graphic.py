# graphic.py
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Enable CUDA optimization
torch.backends.cudnn.benchmark = True

# Settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

BATCH_SIZE = 32
IMAGE_SIZE = 64
CNN_LAYERS = 10
CHANNELS = 3
NUM_CLASSES = 100
EPOCHS = 10000
NUM_SAMPLES = 10**7  # very large; doesn't consume RAM



# Dynamically generated dataset
class OnTheFlyDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = torch.randn(CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
        y = torch.randint(0, NUM_CLASSES, (1,)).item()
        return x, y

# Heavy CNN to stress GPU
class HeavyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        in_channels = CHANNELS
        for _ in range(CNN_LAYERS):  # deep network
            layers.append(nn.Conv2d(in_channels, 128, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            in_channels = 128
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

def train():
    dataset = OnTheFlyDataset(NUM_SAMPLES)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,             # increase if your CPU can handle
        pin_memory=True,
        persistent_workers=True    # keeps workers alive
    )

    model = HeavyCNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for batch_x, batch_y in loader:
            torch.cuda.empty_cache()
            batch_x = batch_x.to(DEVICE, non_blocking=True)
            batch_y = batch_y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {running_loss:.4f}")

if __name__ == "__main__":
    train()
