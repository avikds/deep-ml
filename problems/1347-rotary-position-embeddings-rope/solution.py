import torch
import torch.nn as nn


def build_rope_cache(seq_len, head_dim, base=10000.0):
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )

    positions = torch.arange(seq_len, dtype=torch.float32)
    theta = torch.outer(positions, inv_freq)

    return torch.cos(theta), torch.sin(theta)


def apply_rope(x, cos, sin):
    head_dim = x.shape[-1]
    half_dim = head_dim // 2

    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]

    rotated = torch.cat((-x2, x1), dim=-1)

    cos = torch.cat((cos, cos), dim=-1)
    sin = torch.cat((sin, sin), dim=-1)

    return x * cos + rotated * sin