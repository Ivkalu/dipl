import torch

def mel_loss(output, target, sample_rate):
    from torchaudio.transforms import MelSpectrogram

    mel_transform = MelSpectrogram(sample_rate=sample_rate, n_fft=1024, hop_length=256, n_mels=80)
    mel_output = mel_transform(output)
    mel_target = mel_transform(target)

    return torch.nn.functional.l1_loss(mel_output, mel_target)