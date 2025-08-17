import torch
import torch.nn as nn
from torch import Tensor

# You can use this placeholder for Mamba implementation if no library is installed
# For full-featured Mamba, install from https://github.com/locuslab/mamba

class MambaBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, state_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        
        # simple linear layer to expand input
        self.expand = nn.Linear(input_dim, hidden_dim)
        
        # placeholder SSM kernel (here we just use a Conv1d to mimic SSM behavior)
        self.ssm_kernel = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        
        # projection to output
        self.contract = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.Tanh()
    
    def forward(self, x: Tensor) -> Tensor:
        # x: [batch, seq_len, input_dim]
        x = self.expand(x)                  # [batch, seq_len, hidden_dim]
        x = x.transpose(1, 2)               # [batch, hidden_dim, seq_len]
        x = self.ssm_kernel(x)
        x = x.transpose(1, 2)               # [batch, seq_len, hidden_dim]
        x = self.contract(x)
        x = self.activation(x)
        return x

class ConvMambaModel(nn.Module):
    def __init__(self, conv1d_filters, hidden_units, state_dim=64):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=conv1d_filters,
                               kernel_size=12, padding='same')
        self.conv2 = nn.Conv1d(in_channels=conv1d_filters, out_channels=conv1d_filters,
                               kernel_size=12, padding='same')
        
        self.mamba = MambaBlock(conv1d_filters, hidden_units, state_dim)
        self.fc = nn.Linear(hidden_units, 2)
    
    def forward(self, x: Tensor) -> Tensor:
        # [batch_size, input_size, channels] -> [batch, channels, input_size]
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.transpose(1, 2)
        x = self.mamba(x)
        x = x[:, -1, :]  # take last time step
        x = self.fc(x)
        return x