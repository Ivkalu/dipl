import torch
import torchaudio
import os

class ChunkedWavDataset(torch.utils.data.Dataset):
    def __init__(self, input_wav_paths, output_wav_paths, chunk_size=110250, hop_size=None):
        assert len(input_wav_paths) == len(output_wav_paths)
        self.chunk_size = chunk_size
        self.hop_size = hop_size or chunk_size  # no overlap by default

        self.chunks = []  # list of (file_idx, start_sample)

        self.input_paths = input_wav_paths
        self.output_paths = output_wav_paths

        self.input_lengths = []

        for idx, path in enumerate(input_wav_paths):
            info = torchaudio.info(path)
            total_len = info.num_frames
            self.input_lengths.append(total_len)

            # generate chunk start positions
            starts = list(range(0, total_len - chunk_size + 1, self.hop_size))
            for s in starts:
                self.chunks.append((idx, s))

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, index):
        file_idx, start = self.chunks[index]
        input_wav, sr1 = torchaudio.load(self.input_paths[file_idx], frame_offset=start, num_frames=self.chunk_size)
        output_wav, sr2 = torchaudio.load(self.output_paths[file_idx], frame_offset=start, num_frames=self.chunk_size)

        assert sr1 == sr2, "Sample rates must match"
        
        input_wav = input_wav.reshape(-1, 1)
        output_wav = output_wav.reshape(-1, 1)

        # If very last chunk is shorter than chunk_size
        #input_wav = self._pad_to_chunk(input_wav)
        #output_wav = self._pad_to_chunk(output_wav)

        if input_wav.ndim == 1:
            input_wav = input_wav.unsqueeze(1)
        if output_wav.ndim == 1:
            output_wav = output_wav.unsqueeze(1)


        input_wav = self._pad_or_truncate(input_wav)
        output_wav = self._pad_or_truncate(output_wav)

        return input_wav, output_wav

    def _pad_to_chunk(self, wav):
        if wav.shape[0] < self.chunk_size:
            pad = torch.zeros(self.chunk_size - wav.shape[0], wav.shape[1])
            wav = torch.cat([wav, pad], dim=0)
        return wav
    
    def _pad_or_truncate(self, wav):
        if wav.shape[0] < self.chunk_size:
            pad = torch.zeros(self.chunk_size - wav.shape[0], wav.shape[1])
            wav = torch.cat([wav, pad], dim=0)
        elif wav.shape[0] > self.chunk_size:
            wav = wav[:self.chunk_size]
        return wav

