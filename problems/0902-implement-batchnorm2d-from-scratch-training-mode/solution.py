import torch

def batchnorm2d(x, gamma, beta, eps=1e-5):
    # Mean and variance for each channel over N, H, W
    mean = x.mean(dim=(0, 2, 3), keepdim=True)
    var = x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)

    # Normalize
    x_norm = (x - mean) / torch.sqrt(var + eps)

    # Reshape C -> (1, C, 1, 1) for broadcasting
    gamma = gamma.view(1, -1, 1, 1)
    beta = beta.view(1, -1, 1, 1)

    # Scale and shift
    return gamma * x_norm + beta