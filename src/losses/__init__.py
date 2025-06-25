from spectral import spectral_loss
from mel import mel_loss

total_loss = stft_loss + 0.5 * mel_loss + 0.1 * envelope_loss