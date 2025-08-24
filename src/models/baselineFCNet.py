import torch.nn as nn
import torch

class BaselineFCNet(nn.Module):
    name="baselineFC"

    def __init__(self, channels=2, hidden_size=32, hidden_layers=1):
        super(BaselineFCNet, self).__init__()

        layers = [nn.Linear(channels, hidden_size), nn.ReLU()]

        for _ in range(hidden_layers - 1):  # Add additional hidden layers if needed
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_size, channels))  # Output layer

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # [batch_size, input_size, channels] 
        x = x[:, -1, :]
        out = self.model(x)
        
        # [400, 1, 2]
        return out
    
    def process_whole_audio_file(self, x, y, input_size=100, batch_size=400, device='cpu'):
        """self.eval()
        #[channels, len])

        # inp = x.T.unfold(0, size=input_size, step=1).permute(0, 2, 1).to(device)
        
        predicted = []
        with torch.no_grad():
            for i in range(0, x.shape[1]):
                if not i%100: print(i)
                pred = self(x[:,max(0,i-100):i+1].T.unsqueeze(0).to(device))
                predicted.append(pred)

        predicted_audio = torch.cat(predicted).T
        return predicted_audio
        """

        self.eval()
        with torch.no_grad():
            # Build all sliding windows
            seq_len = x.shape[1]
            num_windows = seq_len
            windows = []

            for i in range(seq_len):
                start_idx = max(0, i - input_size)
                chunk = x[:, start_idx:i+1]
                # Pad to exactly input_size+1 if needed
                if chunk.shape[1] < input_size + 1:
                    pad_width = input_size + 1 - chunk.shape[1]
                    chunk = torch.cat([
                        torch.zeros((x.shape[0], pad_width), dtype=x.dtype),
                        chunk
                    ], dim=1)
                windows.append(chunk.T.unsqueeze(0))

            # Stack into shape [num_windows, input_size+1, channels]
            windows = torch.cat(windows, dim=0)

            # Run in batches
            predicted = []
            for i in range(0, num_windows, batch_size):
                batch = windows[i:i+batch_size].to(device)
                pred = self(batch)
                predicted.append(pred.cpu())

            predicted_audio = torch.cat(predicted, dim=0).T
            return predicted_audio