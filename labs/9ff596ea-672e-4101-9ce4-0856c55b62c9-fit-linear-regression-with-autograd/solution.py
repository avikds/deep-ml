import torch
import numpy as np

def fit_linear_regression(X, y, lr=0.1, steps=500):
    """Fit y ~= X @ w + b with full-batch GD using only autograd."""
    X = X.float()
    y = y.float().reshape(-1)

    n, d = X.shape

    w = torch.zeros(d, dtype=X.dtype, requires_grad=True)
    b = torch.zeros((), dtype=X.dtype, requires_grad=True)

    for _ in range(steps):
        pred = X @ w + b
        loss = torch.mean((pred - y) ** 2)

        loss.backward()

        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad

        w.grad.zero_()
        b.grad.zero_()

    return w.detach(), b.detach()