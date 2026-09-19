import torch
import torch.nn as nn
import torch.optim as optim


class TinyCNN(nn.Module):
    """Small CPU CNN for horizontal/vertical stripe classification."""

    def __init__(self, img_size=8, n_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Two 2x2 pooling operations reduce H,W by 4.
        feature_size = img_size // 4

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * feature_size * feature_size, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_model(img_size=8, n_classes=2):
    """Return a TinyCNN instance."""
    torch.manual_seed(0)
    return TinyCNN(img_size=img_size, n_classes=n_classes)


def train_model(model, train_x, train_y,
                epochs=15, lr=0.01, batch_size=32, seed=0):
    """Train model on CPU and return the trained model."""

    # Deterministic initialization/shuffling behavior.
    torch.manual_seed(seed)

    model.train()

    criterion = nn.CrossEntropyLoss()

    # Adam is effective for this tiny classification problem.
    optimizer = optim.Adam(model.parameters(), lr=lr)

    n = train_x.shape[0]

    # Generator gives deterministic mini-batch permutations.
    generator = torch.Generator()
    generator.manual_seed(seed)

    for epoch in range(epochs):
        # Shuffle indices without moving anything to GPU.
        indices = torch.randperm(n, generator=generator)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_idx = indices[start:end]

            x_batch = train_x[batch_idx]
            y_batch = train_y[batch_idx]

            optimizer.zero_grad()

            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            loss.backward()
            optimizer.step()

    return model