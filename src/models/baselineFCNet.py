import torch.nn as nn

class BaselineFCNet(nn.Module):
    name="baselineFC"

    def __init__(self, input_size=1, hidden_size=32, hidden_layers=1, output_size=1):
        super(BaselineFCNet, self).__init__()

        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]

        for _ in range(hidden_layers - 1):  # Add additional hidden layers if needed
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_size, output_size))  # Output layer

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)

        # Ensure the output has the same length as the input (reshape if necessary)
        return out