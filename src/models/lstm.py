import torch
import torch.nn as nn



# Model klasa
class LSTM(nn.Module):
    def __init__(self, conv1d_filters = 16, hidden_units = 26, seq2seq=False):
        super(LSTM, self).__init__()

        # Conv1D layers (PyTorch uses Conv1d with input shape: [batch, channels, sequence_length])
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=conv1d_filters,
                               kernel_size=12, padding='same')
        self.conv2 = nn.Conv1d(in_channels=conv1d_filters, out_channels=conv1d_filters,
                               kernel_size=12, padding='same')

        self.lstm = nn.LSTM(input_size=conv1d_filters, hidden_size=hidden_units, batch_first=True)
        self.fc = nn.Linear(hidden_units, 2)
        self.seq2seq = seq2seq

        if self.seq2seq:
            torch.backends.cudnn.enabled = False
        else:
            torch.backends.cudnn.enabled = True

    def forward(self, x):
                                    # [batch_size, input_size, channels]
        x = x.transpose(1, 2)       # [batch_size, channels, input_size]
        x = self.conv1(x)           # [batch_size, conv1d_filters, input_size]
        x = self.conv2(x)           # [batch_size, conv1d_filters, input_size]
        x = x.transpose(1, 2)       # [batch_size, input_size, conv1d_filters]
        x, _ = self.lstm(x)         # [batch_size, input_size, hidden_units]

        if not self.seq2seq:
            x = x[:, -1, :]             # [batch_size, hidden_units] TODO remove
        x = self.fc(x)              # [batch_size, channels]
        return x
        
    def save(self):
        #model_path = f'models/{name}/{name}.pt'
        #torch.save({
        #    'model_state_dict': model.state_dict(),
        #    'input_size': input_size,
        #    'conv1d_filters': conv1d_filters,
        #    'conv1d_stride': conv1d_stride,
        #    'hidden_units': hidden_units
        #}, model_path)
        pass