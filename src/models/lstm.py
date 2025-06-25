import torch
import torch.nn as nn

class GuitarPedalLSTM(nn.Module):
    name="lstm"

    def __init__(self, input_size=1, hidden_size=128, num_layers=2, output_size=1, bidirectional=False):
        super(GuitarPedalLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=bidirectional)

        self.fc = nn.Linear(hidden_size * self.num_directions, output_size)

    def forward(self, x):
        # x: (batch_size, sequence_length, input_size)
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))  # out: (batch, seq_len, hidden_size*num_directions)
        out = self.fc(out)               # out: (batch, seq_len, output_size)
        return out
