
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBN(nn.Module):
    def __init__(self, ci, co, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(
            ci, co, 3, stride=stride, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(co)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = ConvBN(c, c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)

    def forward(self, x):
        return F.relu(x + self.bn2(self.c2(self.c1(x))), inplace=True)


class CIFARNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            ConvBN(3, 64),
            ConvBN(64, 128, 2),
            ResBlock(128),
            ConvBN(128, 256, 2),
            ResBlock(256),
            ConvBN(256, 384, 2),
            ResBlock(384),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.fc = nn.Linear(384, 10)

    def forward(self, x):
        return self.fc(self.features(x))


def train(X_train, y_train, budget_seconds):
    t0 = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(42)
    np.random.seed(42)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # Entire training set on the GPU.
    x = torch.as_tensor(
        X_train.transpose(0, 3, 1, 2).copy(),
        dtype=torch.float32,
        device=device
    ) / 255.0

    y = torch.as_tensor(
        y_train, dtype=torch.long, device=device
    )

    mean = torch.tensor(
        [0.4914, 0.4822, 0.4465],
        device=device
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        [0.2470, 0.2435, 0.2616],
        device=device
    ).view(1, 3, 1, 1)

    x = (x - mean) / std

    # Reflect padding allows random crops without CPU augmentation.
    padded = F.pad(x, (4, 4, 4, 4), mode="reflect")
    padded_flat = padded.reshape(len(x), 3, 1600)

    n = len(x)
    batch_size = 256

    model = CIFARNet().to(device)
    model = model.to(memory_format=torch.channels_last)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0025,
        weight_decay=0.0007
    )

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    deadline = t0 + budget_seconds * 0.87
    total_time = max(0.001, budget_seconds * 0.87)

    yy = torch.arange(32, device=device).view(1, 32, 1)
    xx = torch.arange(32, device=device).view(1, 1, 32)

    epoch = 0
    step = 0

    model.train()

    while time.perf_counter() < deadline:
        epoch += 1
        order = torch.randperm(n, device=device)

        for start in range(0, n, batch_size):
            if time.perf_counter() > deadline:
                break

            ids = order[start:start + batch_size]
            b = len(ids)

            # Per-image random crop and horizontal flip.
            y0 = torch.randint(0, 9, (b, 1, 1), device=device)
            x0 = torch.randint(0, 9, (b, 1, 1), device=device)
            flip = torch.rand(b, 1, 1, device=device) < 0.5

            cols = torch.where(flip, 31 - xx, xx)
            yi = y0 + yy
            xi = x0 + cols

            indices = (yi * 40 + xi).reshape(b, 1, 1024)
            indices = indices.expand(-1, 3, -1)

            xb = torch.gather(
                padded_flat[ids], 2, indices
            ).reshape(b, 3, 32, 32)

            # Mild color augmentation.
            brightness = (
                torch.rand(b, 1, 1, 1, device=device) - 0.5
            ) * 0.15

            contrast = (
                0.9 + 0.2 * torch.rand(b, 1, 1, 1, device=device)
            )

            xb = xb * contrast + brightness
            xb = xb.contiguous(memory_format=torch.channels_last)

            yb = y[ids]

            progress = min(
                1.0,
                (time.perf_counter() - t0) / total_time
            )

            warmup = min(1.0, progress / 0.06)
            decay = max(0.0, (progress - 0.06) / 0.94)

            lr = 0.0025 * warmup * (
                0.10 + 0.90 * 0.5 *
                (1.0 + math.cos(math.pi * decay))
            )

            for group in optimizer.param_groups:
                group["lr"] = lr

            optimizer.zero_grad(set_to_none=True)

            # Mixup on a fraction of batches.
            use_mix = bool(torch.rand((), device=device) < 0.25)

            with torch.cuda.amp.autocast(enabled=use_amp):
                if use_mix and b > 1:
                    perm = torch.randperm(b, device=device)
                    lam = 0.6 + 0.4 * torch.rand((), device=device)

                    mixed = lam * xb + (1.0 - lam) * xb[perm]
                    logits = model(mixed)

                    loss = (
                        lam * F.cross_entropy(
                            logits, yb, label_smoothing=0.02
                        )
                        + (1.0 - lam) * F.cross_entropy(
                            logits, yb[perm], label_smoothing=0.02
                        )
                    )
                else:
                    logits = model(xb)
                    loss = F.cross_entropy(
                        logits, yb, label_smoothing=0.02
                    )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            step += 1

        # Synchronize so the actual GPU work is included in the budget.
        if device.type == "cuda":
            torch.cuda.synchronize()

    model.eval()

    return {
        "model": model,
        "device": device,
        "mean": mean,
        "std": std,
        "epochs": epoch
    }


def predict_proba(model, X):
    net = model["model"]
    device = model["device"]
    mean = model["mean"]
    std = model["std"]

    x = torch.as_tensor(
        X.transpose(0, 3, 1, 2).copy(),
        dtype=torch.float32,
        device=device
    ) / 255.0

    x = (x - mean) / std
    net.eval()

    outputs = []

    with torch.no_grad():
        for start in range(0, len(x), 256):
            xb = x[start:start + 256]
            xb = xb.contiguous(memory_format=torch.channels_last)

            with torch.cuda.amp.autocast(
                enabled=device.type == "cuda"
            ):
                logits1 = net(xb)
                logits2 = net(torch.flip(xb, dims=[3]))

                probs1 = torch.softmax(logits1.float(), dim=1)
                probs2 = torch.softmax(logits2.float(), dim=1)

                probs = 0.5 * (probs1 + probs2)

            outputs.append(probs.cpu().numpy())

    if not outputs:
        return np.empty((0, 10), dtype=np.float32)

    P = np.concatenate(outputs, axis=0)
    P = np.maximum(P, 1e-8)
    P /= P.sum(axis=1, keepdims=True)

    return P.astype(np.float32)
