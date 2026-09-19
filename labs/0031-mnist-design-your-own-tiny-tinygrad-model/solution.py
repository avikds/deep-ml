from tinygrad import Tensor, nn

def build_model():
    class TinyNet:
        def __init__(self):
            # 32 filters × 1 input channel × 3 × 3 = 288 params
            self.conv = nn.Conv2d(1, 32, 3, padding=1)

            # 32 channels × 4 spatial regions × 10 classes
            # = 1280 params
            # Total = 1568 trainable parameters
            self.fc = nn.Linear(32 * 4, 10)

        def __call__(self, x: Tensor) -> Tensor:
            x = self.conv(x).relu()

            # Preserve coarse spatial information by pooling each
            # feature map over four quadrants.
            #
            # Input is 28×28, so each quadrant is 14×14.
            q1 = x[:, :, :14, :14].mean(axis=(2, 3))
            q2 = x[:, :, :14, 14:].mean(axis=(2, 3))
            q3 = x[:, :, 14:, :14].mean(axis=(2, 3))
            q4 = x[:, :, 14:, 14:].mean(axis=(2, 3))

            # (N, 32, 4) -> (N, 128)
            x = Tensor.cat(q1, q2, q3, q4, dim=1)

            return self.fc(x)

    return TinyNet()