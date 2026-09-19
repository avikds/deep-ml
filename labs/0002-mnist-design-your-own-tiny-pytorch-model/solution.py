import torch
import torch.nn as nn

def build_model() -> nn.Module:
    class TinyNet(nn.Module):
        def __init__(self):
            super().__init__()

            self.features = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=5, padding=2, bias=True),   # 416
                nn.BatchNorm2d(16),                                      # 32
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),

                # Depthwise conv
                nn.Conv2d(16, 16, kernel_size=3, padding=1,
                          groups=16, bias=True),                        # 160
                nn.Conv2d(16, 16, kernel_size=1, bias=True),             # 272
                nn.BatchNorm2d(16),                                      # 32
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),

                # Second depthwise-separable block
                nn.Conv2d(16, 16, kernel_size=3, padding=1,
                          groups=16, bias=True),                        # 160
                nn.Conv2d(16, 16, kernel_size=1, bias=True),             # 272
                nn.BatchNorm2d(16),                                      # 32
                nn.ReLU(inplace=True),

                # Keep a little spatial information
                nn.AdaptiveAvgPool2d((2, 2)),
            )

            self.classifier = nn.Linear(16 * 2 * 2, 10)                 # 650

        def forward(self, x):
            # Supports either [N,28,28] or [N,1,28,28]
            if x.dim() == 3:
                x = x.unsqueeze(1)

            x = x.float()

            # Robust to either [0,255] or [0,1] input.
            if x.max() > 1.5:
                x = x / 255.0

            x = self.features(x)
            x = torch.flatten(x, 1)
            return self.classifier(x)

    return TinyNet()