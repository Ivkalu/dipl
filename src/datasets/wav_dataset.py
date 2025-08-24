import os
import math
import argparse
import numpy as np
from scipy.io import wavfile
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import IPython.display as ipd
from IPython.display import display, Audio
from bisect import bisect_left
import torchaudio


# Normalize funkcija
def normalize(data: torch.Tensor) -> torch.Tensor:
    data_max = data.max()
    data_min = data.min()
    data_norm = torch.maximum(data_max.abs(), data_min.abs())
    return data / (data_norm + 1e-8)


from collections import defaultdict, deque
import time

from collections import defaultdict, deque
from collections.abc import MutableMapping
from typing import TypeVar, Generic, Optional, Deque
import time


K = TypeVar('K')
V = TypeVar('V')

class FrequencyQueue(Generic[K, V], MutableMapping[K, V]):
    def __init__(self, capacity: int):
        self.capacity: int = capacity
        self.freq: defaultdict[K, int] = defaultdict(int)
        self.recency: dict[K, float] = {}
        self.data: Deque[K] = deque()
        self.store: dict[K, V] = {}

    def _sort_queue(self) -> None:
        self.data = deque(
            sorted(
                self.data,
                key=lambda k: (-self.freq[k], -self.recency[k])
            )
        )

    def _touch(self, key: K) -> None:
        self.freq[key] += 1
        self.recency[key] = time.time()
        if key in self.data:
            self.data.remove(key)
        self.data.append(key)
        self._sort_queue()

    def __getitem__(self, key: K) -> V:
        if key in self.store:
            self._touch(key)
            return self.store[key]
        raise KeyError(f"{key} not found")

    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        if key in self.store:
            self._touch(key)
            return self.store[key]
        return default

    def __setitem__(self, key: K, value: V) -> None:
        if key not in self.store and len(self.data) >= self.capacity:
            oldest = self.data.pop()
            del self.store[oldest]
            del self.freq[oldest]
            del self.recency[oldest]

        self.store[key] = value
        self._touch(key)

    def __contains__(self, key: object) -> bool:
        if key in self.store:
            self._touch(key)  # type: ignore
            return True
        return False

    def __delitem__(self, key: K) -> None:
        if key in self.store:
            self.data.remove(key)
            del self.store[key]
            del self.freq[key]
            del self.recency[key]
        else:
            raise KeyError(f"{key} not found")

    def __iter__(self):
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.store)

    def __repr__(self) -> str:
        return f"Queue: {[f'{k}:{self.store[k]}' for k in self.data]}"



class FrequencyQueueDataset(Dataset):
    

    def __init__(self, input_dir, target_dir, input_size, max_samples, max_wav_files):
        self.input_size = input_size
        self.pairs = []
        self.cum_sum = []

        
        self.file_indexes_ready = FrequencyQueue(max_wav_files)

        input_files = sorted(os.listdir(input_dir))
        
        total_length = 0
        for fname in input_files:
            # if file is not audio, ignore
            if not fname.lower().endswith('.wav'):
              continue
            # if there is no matching output file, ignore
            in_path = os.path.join(input_dir, fname)
            out_path = os.path.join(target_dir, fname)
            if not os.path.exists(out_path):
                continue

            # if samplerates don't match, ignore
            in_rate, in_data = wavfile.read(in_path)
            out_rate, out_data = wavfile.read(out_path)

            if in_data.ndim != 2 or out_data.ndim != 2:
                continue

            if in_rate != 44100 or out_rate != 44100:
                continue

            max_len = max(len(in_data), len(out_data)) #max lenth in number of samples
            # in data cant be longer than out data

            if max_samples is not None and total_length + max_len > max_samples:
                self.cum_sum.append(max_samples - 1)
                self.pairs.append((in_path, out_path))
                break

            total_length += max_len

            self.cum_sum.append(total_length)
            self.pairs.append((in_path, out_path))

        assert len(self.pairs) > 0
        print(f"Length in seconds: {len(self) / 44100:.2f}, Num samples: {len(self)}")
        print(f"Total files used in frequency dataset: {len(self.pairs)}\nMaximum files that can be loaded at the same time: {max_wav_files}")

    def __len__(self):
        return self.cum_sum[-1] - self.input_size + 1

    def __getitem__(self, idx):
        # Find which file the idx corresponds to

        file_index_start = bisect_left(self.cum_sum, idx)
        file_index_end = bisect_left(self.cum_sum, idx + self.input_size)

        samples_input_list = []
        samples_output_list = []

        for file_idx in range(file_index_start, file_index_end + 1):
            if file_idx in self.file_indexes_ready:
                in_data, out_data = self.file_indexes_ready[file_idx]
            else:
                in_path, out_path = self.pairs[file_idx]
                in_data, in_rate = torchaudio.load(in_path) 
                out_data, out_rate = torchaudio.load(out_path)

                #in_data = normalize(in_data)
                #out_data = normalize(out_data)

                max_len = max(in_data.shape[1], out_data.shape[1])
                if in_data.shape[0] < max_len:
                    in_data = torch.nn.functional.pad(in_data, (0, max_len - in_data.shape[1]))
                if out_data.shape[0] < max_len:
                    out_data = torch.nn.functional.pad(out_data, (0, max_len - out_data.shape[1]))

                max_allowed_len = self.cum_sum[file_idx] - (self.cum_sum[file_idx - 1] if file_idx > 0 else 0)
                if in_data.shape[1] > max_allowed_len:
                    in_data = in_data[:, :max_allowed_len]
                if out_data.shape[1] > max_allowed_len:
                    out_data = out_data[:, :max_allowed_len]

                self.file_indexes_ready[file_idx] = (in_data, out_data)

            # update self.file_indexes_ready
            samples_input_list.append(in_data)
            samples_output_list.append(out_data)

        samples_input = torch.cat(samples_input_list, dim=1) # [2, samples]
        samples_output = torch.cat(samples_output_list, dim=1) # [2, samples]

        offset = self.cum_sum[file_index_start - 1] if file_index_start > 0 else 0
        local_idx = idx - offset

        x_window = samples_input[:, local_idx : local_idx + self.input_size]  # shape: [2, input_size]
        y_value = samples_output[:, local_idx + self.input_size - 1]        # shape: [2]

        x_tensor = x_window.transpose(0, 1)
        y_tensor = y_value

        return x_tensor, y_tensor
    




class WaveformFileDataset(Dataset):
    
    def __init__(self, input_dir, target_dir):
        self.pairs = []

        input_files = sorted(os.listdir(input_dir))
        
        for fname in input_files:
            # if file is not audio, ignore
            if not fname.lower().endswith('.wav'):
              continue
            # if there is no matching output file, ignore
            in_path = os.path.join(input_dir, fname)
            out_path = os.path.join(target_dir, fname)
            if not os.path.exists(out_path):
                continue

            # if samplerates don't match, ignore
            in_rate, in_data = wavfile.read(in_path)
            out_rate, out_data = wavfile.read(out_path)

            if in_data.ndim != 2 or out_data.ndim != 2:
                continue

            if in_rate != 44100 or out_rate != 44100:
                continue

            self.pairs.append((in_path, out_path))


    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        # Find which file the idx corresponds to
        in_path, out_path = self.pairs[idx]
        in_data, in_rate = torchaudio.load(in_path) 
        out_data, out_rate = torchaudio.load(out_path)

        #in_data = normalize(in_data)
        #out_data = normalize(out_data)

        max_len = max(in_data.shape[1], out_data.shape[1])
        if in_data.shape[0] < max_len:
            in_data = torch.nn.functional.pad(in_data, (0, max_len - in_data.shape[1]))
        if out_data.shape[0] < max_len:
            out_data = torch.nn.functional.pad(out_data, (0, max_len - out_data.shape[1]))

        return in_data, out_data
    

