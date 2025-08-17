import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchaudio
import glob
import os

# --------------------------
# Hyperparameters
# --------------------------
BATCH_SIZE = 16
LR = 0.00005
EPOCHS = 100
LATENT_DIM = 100
AUDIO_LENGTH = 16384  # number of audio samples per clip

# --------------------------
# Audio Dataset
# --------------------------
class AudioDataset(Dataset):
    def __init__(self, folder_path):
        self.files = glob.glob(os.path.join(folder_path, "*.wav"))
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        waveform, sr = torchaudio.load(self.files[idx])
        waveform = waveform.mean(dim=0)  # Convert to mono
        if waveform.size(0) < AUDIO_LENGTH:
            waveform = torch.nn.functional.pad(waveform, (0, AUDIO_LENGTH - waveform.size(0)))
        else:
            waveform = waveform[:AUDIO_LENGTH]
        waveform = (waveform - waveform.min()) / (waveform.max() - waveform.min())  # Normalize [0,1]
        
        waveform = waveform * 2 - 1  # Scale to [-1,1]
        return waveform.unsqueeze(0)  # shape: [1, AUDIO_LENGTH]

# --------------------------
# Generator
# --------------------------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(LATENT_DIM, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, AUDIO_LENGTH),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.model(z).unsqueeze(1)  # shape: [B,1,AUDIO_LENGTH]

# --------------------------
# Discriminator
# --------------------------
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(AUDIO_LENGTH, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1)  # No Sigmoid
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)
# Then change loss:


# --------------------------
# Training Setup
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = AudioDataset("E:\\source\\dipl\\data\\train\\x\\other")
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

G = Generator().to(device)
D = Discriminator().to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer_G = optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))

# --------------------------
# Training Loop
# --------------------------
for epoch in range(EPOCHS):
    for i, real_audio in enumerate(dataloader):
        real_audio = real_audio.to(device)
        batch_size = real_audio.size(0)

        # Train Discriminator
        z = torch.randn(batch_size, LATENT_DIM).to(device)
        fake_audio = G(z)

        D_real = D(real_audio)
        D_fake = D(fake_audio.detach())

        loss_D = criterion(D_real, torch.ones_like(D_real)) + criterion(D_fake, torch.zeros_like(D_fake))
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # Train Generator
        D_fake = D(fake_audio)
        loss_G = criterion(D_fake, torch.ones_like(D_fake))
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

    print(f"Epoch [{epoch+1}/{EPOCHS}]  Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")

# --------------------------
# Generate Audio
# --------------------------
z = torch.randn(1, LATENT_DIM).to(device)
generated_audio = G(z).squeeze().detach().cpu()
torchaudio.save("generated.wav", generated_audio.unsqueeze(0), 16000)