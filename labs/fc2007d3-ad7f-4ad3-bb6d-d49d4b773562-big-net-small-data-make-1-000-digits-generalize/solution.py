import torch
import torch.nn.functional as F
import math


def train(X, y):
    """Train a regularized 784-800-800-10 MNIST network."""

    X = X.float()
    y = y.long()
    device = X.device

    mean = 0.1307
    std = 0.3081

    net = torch.nn.Sequential(
        torch.nn.Linear(784, 800),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.20),

        torch.nn.Linear(800, 800),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.30),

        torch.nn.Linear(800, 10)
    ).to(device)

    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=0.001,
        weight_decay=0.0003
    )

    epochs = 100
    batch_size = 100

    images = X.reshape(-1, 1, 28, 28)

    ema = None
    ema_decay = 0.95

    def augment(batch):
        n = batch.shape[0]

        # Gentle rotation, translation and scaling.
        angle = (torch.rand(n, device=device) - 0.5) * 0.22
        scale = 0.95 + 0.10 * torch.rand(n, device=device)

        tx = (torch.rand(n, device=device) - 0.5) * 0.12
        ty = (torch.rand(n, device=device) - 0.5) * 0.12

        # Keep some images unchanged.
        active = (torch.rand(n, device=device) > 0.20).float()

        angle = angle * active
        scale = 1.0 + (scale - 1.0) * active
        tx = tx * active
        ty = ty * active

        c = torch.cos(angle) * scale
        s = torch.sin(angle) * scale

        theta = torch.zeros(n, 2, 3, device=device)

        theta[:, 0, 0] = c
        theta[:, 0, 1] = -s
        theta[:, 1, 0] = s
        theta[:, 1, 1] = c

        theta[:, 0, 2] = tx
        theta[:, 1, 2] = ty

        grid = F.affine_grid(
            theta,
            batch.shape,
            align_corners=False
        )

        return F.grid_sample(
            batch,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False
        )

    for epoch in range(epochs):

        # Cosine learning-rate decay.
        progress = epoch / epochs

        lr = 0.0001 + 0.0009 * (
            1.0 + math.cos(math.pi * progress)
        ) / 2.0

        for group in optimizer.param_groups:
            group["lr"] = lr

        net.train()

        permutation = torch.randperm(
            len(X),
            device=device
        )

        for start in range(0, len(X), batch_size):
            idx = permutation[start:start + batch_size]

            xb = augment(images[idx])
            yb = y[idx]

            xb = xb.reshape(-1, 784)
            xb = (xb - mean) / std

            logits = net(xb)

            loss = F.cross_entropy(
                logits,
                yb,
                label_smoothing=0.02
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                net.parameters(),
                5.0
            )

            optimizer.step()

            # Max-norm constraint on hidden-layer incoming weights.
            with torch.no_grad():
                for layer in (net[0], net[3]):
                    norms = layer.weight.norm(
                        dim=1,
                        keepdim=True
                    ).clamp_min(1e-8)

                    layer.weight.mul_(
                        (4.0 / norms).clamp(max=1.0)
                    )

        # Start EMA after the initial learning phase.
        if epoch >= 20:
            with torch.no_grad():
                current = list(net.parameters())

                if ema is None:
                    ema = [
                        p.detach().clone()
                        for p in current
                    ]
                else:
                    for avg, p in zip(ema, current):
                        avg.mul_(ema_decay).add_(
                            p.detach(),
                            alpha=1.0 - ema_decay
                        )

    # Use the averaged parameters.
    with torch.no_grad():
        if ema is None:
            params = [
                p.detach().clone()
                for p in net.parameters()
            ]
        else:
            params = ema

        W1, b1, W2, b2, W3, b3 = params

        # IMPORTANT: Adjust bias BEFORE scaling W1.
        b1 = b1 - (mean / std) * W1.sum(dim=1)

        # Fold input normalization into the first layer.
        W1 = W1 / std

    return {
        "W1": W1.float(),
        "b1": b1.float(),
        "W2": W2.float(),
        "b2": b2.float(),
        "W3": W3.float(),
        "b3": b3.float()
    }