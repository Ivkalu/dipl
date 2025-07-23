import torch
import torchaudio

class WavDataset(torch.utils.data.Dataset):
    def __init__(self, input_wav_paths, output_wav_paths, max_length=110250):
        self.input_wav_paths = input_wav_paths
        self.output_wav_paths = output_wav_paths
        self.max_length = max_length

    def __len__(self):
        return len(self.input_wav_paths)

    def __getitem__(self, idx):
        input_wav, sr1 = torchaudio.load(self.input_wav_paths[idx])
        output_wav, sr2 = torchaudio.load(self.output_wav_paths[idx])

        assert sr1 == sr2, "Sample rates must match!"

        input_wav = input_wav.reshape(-1, 1)
        output_wav = output_wav.reshape(-1, 1)

        #if self.max_length is not None:
        #    input_wav = self._pad_or_truncate(input_wav)
        #    output_wav = self._pad_or_truncate(output_wav)

        return input_wav, output_wav

    def _pad_or_truncate(self, wav_tensor):
        """Pads with zeros or truncates to `max_length`."""
        length = wav_tensor.shape[0]

        if length > self.max_length:
            return wav_tensor[:self.max_length]

        elif length < self.max_length:
            padding = torch.zeros(self.max_length - length, 1)
            return torch.cat((wav_tensor, padding), dim=0)

        return wav_tensor