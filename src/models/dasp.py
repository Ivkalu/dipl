import os
import torch
import torch.nn as nn
import torchaudio
import dasp_pytorch.functional as dsp

# -----------------------------
# Differentiable Pedal Chain
# -----------------------------
class DifferentiablePedalChain(nn.Module):
    def __init__(self, sr=48000):
        super().__init__()
        self.sr = sr

        # Distortion
        self.drive = nn.Parameter(torch.tensor([[[5.0]]]))  # dB

        # Compressor
        self.threshold = nn.Parameter(torch.tensor([[[-20.0]]]))  # dB
        self.ratio = nn.Parameter(torch.tensor([[[4.0]]]))
        self.attack_ms = nn.Parameter(torch.tensor([[[10.0]]]))
        self.release_ms = nn.Parameter(torch.tensor([[[50.0]]]))
        self.knee_db = nn.Parameter(torch.tensor([[[6.0]]]))
        self.makeup_gain_db = nn.Parameter(torch.tensor([[[0.0]]]))

        # Reverb
        self.mix = nn.Parameter(torch.tensor([[[0.3]]]))

    def forward(self, x):
        bs, chs, _ = x.shape

        # Distortion
        x = dsp.distortion(x, self.sr, self.drive.expand(bs, chs, 1))

        # Compressor
        x = dsp.compressor(
            x, self.sr,
            self.threshold.expand(bs, chs, 1),
            self.ratio.expand(bs, chs, 1),
            self.attack_ms.expand(bs, chs, 1),
            self.release_ms.expand(bs, chs, 1),
            self.knee_db.expand(bs, chs, 1),
            self.makeup_gain_db.expand(bs, chs, 1),
        )

        # Reverb
        x = dsp.noise_shaped_reverberation(
            x, self.sr,
            band0_gain=torch.tensor([0.0]), band1_gain=torch.tensor([0.0]),
            band2_gain=torch.tensor([0.0]), band3_gain=torch.tensor([0.0]),
            band4_gain=torch.tensor([0.0]), band5_gain=torch.tensor([0.0]),
            band6_gain=torch.tensor([0.0]), band7_gain=torch.tensor([0.0]),
            band8_gain=torch.tensor([0.0]), band9_gain=torch.tensor([0.0]),
            band10_gain=torch.tensor([0.0]), band11_gain=torch.tensor([0.0]),
            band0_decay=torch.tensor([0.5]), band1_decay=torch.tensor([0.5]),
            band2_decay=torch.tensor([0.5]), band3_decay=torch.tensor([0.5]),
            band4_decay=torch.tensor([0.5]), band5_decay=torch.tensor([0.5]),
            band6_decay=torch.tensor([0.5]), band7_decay=torch.tensor([0.5]),
            band8_decay=torch.tensor([0.5]), band9_decay=torch.tensor([0.5]),
            band10_decay=torch.tensor([0.5]), band11_decay=torch.tensor([0.5]),
            mix=self.mix.expand(bs, chs, 1)
        )
        return x


# -----------------------------
# Training Script (first file only)
# -----------------------------
def train_one_file(x_path, y_path, sr=44100, n_iters=2500, lr=0.01):
    # Load input
    x, sr_x = torchaudio.load(x_path)
    if x.size(0) > 1:  # stereo → mono
        x = torch.mean(x, dim=0, keepdim=True)

    # Load target
    y, sr_y = torchaudio.load(y_path)
    if y.size(0) > 1:  # stereo → mono
        y = torch.mean(y, dim=0, keepdim=True)

    assert sr_x == sr_y == sr, f"Sample rates don't match: {sr_x}, {sr_y}, {sr}"

    # Add batch dimension
    x = x.unsqueeze(0)  # (1, 1, samples)
    y = y.unsqueeze(0)  # (1, 1, samples)

    # Create model
    model = DifferentiablePedalChain(sr=sr)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for n in range(n_iters):
        y_hat = model(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        print(
            f"Step {n+1}/{n_iters}, Loss: {loss.item():.3e}, "
            f"Drive: {model.drive.item():.3f}, Mix: {model.mix.item():.3f}"
        )

    return model


if __name__ == "__main__":
    model = train_one_file(
        x_path="E:\\source\\dipl\\data\\train\\x\\guitar\\1-E1-Major 00.wav",
        y_path="E:\\source\\dipl\\data\\train\\y\\guitar\\simpleDist\\1-E1-Major 00.wav",
        n_iters=1000,
        lr=0.001
    )
    torch.save(model.state_dict(), "pedalchain_model_onefile.pth")
    print("Training complete. Model saved.")
