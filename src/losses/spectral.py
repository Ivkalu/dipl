def spectral_loss(output, target, sample_rate):
    import torch
    import torchaudio

    # Choose multiple window sizes (resolutions)
    resolutions = [(2048, 512), (1024, 256), (512, 128)]  # (win_size, hop_size)
    loss = 0

    for win_size, hop_size in resolutions:
        output_stft = torch.stft(output, win_length=win_size, hop_length=hop_size,
                                 n_fft=win_size, return_complex=True)
        target_stft = torch.stft(target, win_length=win_size, hop_length=hop_size,
                                 n_fft=win_size, return_complex=True)
        loss += torch.nn.functional.l1_loss(torch.abs(output_stft), torch.abs(target_stft))

    return loss / len(resolutions)
