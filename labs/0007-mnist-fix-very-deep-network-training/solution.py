import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()

        self.fc = nn.Linear(hidden_size, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.activation = nn.ReLU()

        # Helps stabilize a very deep residual stack
        self.residual_scale = 0.1

    def forward(self, x):
        residual = x

        out = self.fc(x)
        out = self.bn(out)
        out = self.activation(out)

        return residual + self.residual_scale * out


class DeepNetwork(nn.Module):
    '''
    Deep MNIST network with 30 residual layers.

    The depth requirement is preserved:
      - 1 input projection
      - 30 hidden residual layers
      - 1 output layer

    Residual connections + BatchNorm solve the main gradient-flow
    problem of the original plain 30-layer network.
    '''

    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super().__init__()

        # Input projection
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )

        # Exactly 30 deep layers
        self.layers = nn.ModuleList([
            ResidualBlock(hidden_size) for _ in range(30)
        ])

        # Output classifier
        self.output_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)

        # Project 784 -> 128
        x = self.input_layer(x)

        # 30 residual layers
        for layer in self.layers:
            x = layer(x)

        # Classification logits
        return self.output_layer(x)