import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        self.pad = (kernel_size - 1) * dilation
        super().__init__(in_channels, out_channels, kernel_size,
                         padding=0, dilation=dilation)

    def forward(self, x):
        if self.pad > 0:
            x = F.pad(x, (self.pad, 0))  # Pad only on the left
        return super().forward(x)


class WaveNetBlock(nn.Module):
    def __init__(self, in_channels, res_channels, skip_channels, kernel_size=2, dilation=1):
        super().__init__()
        self.causal_conv = CausalConv1d(in_channels, res_channels, kernel_size, dilation)

        self.conv_filter = nn.Conv1d(res_channels, res_channels, 1)
        self.conv_gate = nn.Conv1d(res_channels, res_channels, 1)

        self.res_conv = nn.Conv1d(res_channels, in_channels, 1)
        self.skip_conv = nn.Conv1d(res_channels, skip_channels, 1)

    def forward(self, x):
        x_input = x
        x = self.causal_conv(x)

        filter_out = torch.tanh(self.conv_filter(x))
        gate_out = torch.sigmoid(self.conv_gate(x))
        x = filter_out * gate_out
        skip = self.skip_conv(x)
        res = self.res_conv(x)

        # Fix for residual connection
        if res.shape[-1] != x_input.shape[-1]:
            x_input = x_input[..., -res.shape[-1]:]
        res = res + x_input

        return res, skip

class WaveNetModel(nn.Module):
    name="wavenet"

    def __init__(self, in_channels=1, res_channels=32, skip_channels=64,
                 num_blocks=3, num_layers=4, kernel_size=2):
        super().__init__()

        self.input_conv = CausalConv1d(in_channels, res_channels, kernel_size=1)
        
        self.blocks = nn.ModuleList()
        for b in range(num_blocks):
            for i in range(num_layers):
                dilation = 2 ** i
                self.blocks.append(
                    WaveNetBlock(res_channels, res_channels, skip_channels, kernel_size, dilation)
                )

        self.relu = nn.ReLU()
        self.output_conv1 = nn.Conv1d(skip_channels, skip_channels, 1)
        self.output_conv2 = nn.Conv1d(skip_channels, in_channels, 1)  # output same shape as input

    def forward(self, x):
        # x: (B, T, C) → transpose to (B, C, T)
        # x: (B, T, C) [4, 110250, 1]
        x = x.transpose(1, 2)
        # x: (B, C, T) [4, 1, 110250]
        x = self.input_conv(x)
        # x: (B, C, T) [4, 64, 0]
        skip_total = 0
        i = 0
        for block in self.blocks:
            i += 1
            x, skip = block(x) # breaks on the i = 11
            skip_total = skip_total + skip

    
        out = self.relu(skip_total)
        out = self.output_conv1(out)
        out = self.relu(out)
        out = self.output_conv2(out)
        # output: (B, C, T) → transpose back
        return out.transpose(1, 2)