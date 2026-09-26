import numpy as np

def unet_time_embedding(
    timesteps: list,
    embed_dim: int,
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
    max_period: int = 10000
) -> np.ndarray:
    """
    Compute time embeddings for a diffusion model U-Net.
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    timesteps = np.asarray(timesteps, dtype=float)
    half_dim = embed_dim // 2

    # Geometrically spaced frequencies.
    i = np.arange(half_dim, dtype=float)
    freqs = np.exp(-np.log(max_period) * i / half_dim)

    # Shape: (B, half_dim)
    args = timesteps[:, None] * freqs[None, :]

    # Sinusoidal embedding: [sin, cos].
    emb = np.concatenate(
        [np.sin(args), np.cos(args)],
        axis=1
    )

    # Two-layer MLP with SiLU activation.
    h = emb @ W1 + b1
    h = h / (1.0 + np.exp(-h))
    h = h @ W2 + b2

    return h