import numpy as np

def per_layer_embedding(
    input_ids,
    token_embed,
    ple_embed,
    proj_weight,
    rms_weight,
    d_ple,
    n_layers,
    eps=1e-6
):
    """
    Compute per-layer embeddings by fusing a projected main embedding
    (after RMSNorm) with a scaled per-layer embedding lookup.

    Returns a nested Python list of shape (B, T, n_layers, d_ple).
    """
    input_ids = np.asarray(input_ids, dtype=int)
    token_embed = np.asarray(token_embed, dtype=float)
    ple_embed = np.asarray(ple_embed, dtype=float)
    proj_weight = np.asarray(proj_weight, dtype=float)
    rms_weight = np.asarray(rms_weight, dtype=float)

    # 1. Main token embedding lookup.
    main = token_embed[input_ids]

    # 2. Per-layer embedding lookup and scaling.
    ple = ple_embed[input_ids] * np.sqrt(d_ple)

    # 3. Project main embedding into per-layer embedding space.
    proj = main @ proj_weight

    # 4. RMSNorm over the last dimension, then scale.
    rms = np.sqrt(np.mean(proj ** 2, axis=-1, keepdims=True) + eps)
    proj_norm = (proj / rms) * rms_weight

    # 5. Fuse both streams.
    combined = (proj_norm + ple) * (2.0 ** -0.5)

    # 6. Split the final dimension into layers and per-layer width.
    B, T = input_ids.shape
    combined = combined.reshape(B, T, n_layers, d_ple)

    return combined.tolist()