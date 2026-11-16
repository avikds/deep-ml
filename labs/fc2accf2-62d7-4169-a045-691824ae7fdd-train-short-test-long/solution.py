import torch
import torch.nn as nn


class Positions(nn.Module):
    def __init__(self, d_model, n_heads, head_dim, n_layers, max_len):
        super().__init__()

        self.n_heads = n_heads

        # Standard ALiBi slopes for 4 attention heads.
        slopes = torch.tensor(
            [0.25, 0.0625, 0.015625, 0.00390625],
            dtype=torch.float32
        )
        self.register_buffer("slopes", slopes)

    def embed(self, x, pos):
        # No absolute position embeddings.
        return x

    def rotate(self, q, k, pos, layer):
        # No rotary transformation; position enters through ALiBi.
        return q, k

    def bias(self, pos, layer):
        # distance[i, j] = i - j
        # Causal attention will mask future positions separately.
        pos = pos.to(dtype=torch.float32)
        distance = pos[:, None] - pos[None, :]

        # More distant past tokens receive a larger negative bias.
        return -self.slopes[:, None, None] * distance[None, :, :]