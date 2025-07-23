import torch

def mel_loss(output, target, sample_rate=44100):
    from torchaudio.transforms import MelSpectrogram

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mel_transform = MelSpectrogram(sample_rate=sample_rate, n_fft=1024, hop_length=256, n_mels=80).to(device)
    mel_output = mel_transform(output)
    mel_target = mel_transform(target)

    return torch.nn.functional.l1_loss(mel_output, mel_target)