import numpy as np

def irope_attention(
    Q: list,
    K: list,
    V: list,
    positions: list,
    layer_index: int,
    rope_layers: list,
    base: float = 10000.0
) -> dict:
    Q = np.asarray(Q, dtype=float)
    K = np.asarray(K, dtype=float)
    V = np.asarray(V, dtype=float)
    positions = np.asarray(positions, dtype=float)

    uses_rope = layer_index in rope_layers

    Q_out = Q.copy()
    K_out = K.copy()

    if uses_rope:
        seq_len, d_head = Q.shape

        if d_head % 2 != 0:
            raise ValueError("d_head must be even for RoPE")

        # Frequencies for each pair of dimensions
        inv_freq = 1.0 / (
            base ** (np.arange(0, d_head, 2, dtype=float) / d_head)
        )

        angles = positions[:, None] * inv_freq[None, :]

        cos_vals = np.cos(angles)
        sin_vals = np.sin(angles)

        q_even = Q[:, 0::2]
        q_odd = Q[:, 1::2]

        k_even = K[:, 0::2]
        k_odd = K[:, 1::2]

        Q_out[:, 0::2] = q_even * cos_vals - q_odd * sin_vals
        Q_out[:, 1::2] = q_even * sin_vals + q_odd * cos_vals

        K_out[:, 0::2] = k_even * cos_vals - k_odd * sin_vals
        K_out[:, 1::2] = k_even * sin_vals + k_odd * cos_vals

    # Scaled dot-product attention
    d_head = Q.shape[1]
    scores = (Q_out @ K_out.T) / np.sqrt(d_head)

    # Stable softmax
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores)
    attention_weights = exp_scores / np.sum(
        exp_scores, axis=-1, keepdims=True
    )

    output = attention_weights @ V

    return {
        "output": np.round(output, 4).tolist(),
        "attention_weights": np.round(attention_weights, 4).tolist(),
        "uses_rope": bool(uses_rope)
    }