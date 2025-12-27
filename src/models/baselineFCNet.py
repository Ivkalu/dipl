import torch.nn as nn

class FCNet(nn.Module):
    name="baselineFC"

    def __init__(self, channels=2, hidden_size=32, hidden_layers=1, seq2seq=False):
        super(FCNet, self).__init__()

        layers = [nn.Linear(channels, hidden_size), nn.ReLU()]

        for _ in range(hidden_layers - 1):  # Add additional hidden layers if needed
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_size, channels))  # Output layer

        self.model = nn.Sequential(*layers)
        self.seq2seq = seq2seq

    def forward(self, x):
        # [batch_size, input_size, channels] 
        if not self.seq2seq: x = x[:, -1, :]
        out = self.model(x)
        
        # [400, 1, 2]
        return out