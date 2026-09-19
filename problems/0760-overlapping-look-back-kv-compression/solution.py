import numpy as np

def lookback_compressed_kv(H, W_aKV, W_bKV, W_aZ, W_bZ, B_a, B_b, m):
    """
    H: (n, d) hidden states; n must be divisible by m
    W_aKV, W_bKV: (d, c) KV projection weights
    W_aZ, W_bZ: (d, c) compression-weight projection matrices
    B_a, B_b: (m, c) learnable positional biases
    m: block size

    Returns:
        list of lists of shape (n // m, c)
    """
    H = np.asarray(H, dtype=float)
    W_aKV = np.asarray(W_aKV, dtype=float)
    W_bKV = np.asarray(W_bKV, dtype=float)
    W_aZ = np.asarray(W_aZ, dtype=float)
    W_bZ = np.asarray(W_bZ, dtype=float)
    B_a = np.asarray(B_a, dtype=float)
    B_b = np.asarray(B_b, dtype=float)

    n = H.shape[0]

    if n % m != 0:
        raise ValueError("n must be divisible by m")

    num_blocks = n // m

    # Project all tokens once.
    a_kv = H @ W_aKV
    b_kv = H @ W_bKV
    a_z = H @ W_aZ
    b_z = H @ W_bZ

    outputs = []

    for i in range(num_blocks):
        start = i * m
        end = start + m

        # Current-block stream a.
        kv_a = a_kv[start:end]
        z_a = a_z[start:end] + B_a

        if i == 0:
            # No look-back block: its logits are -inf, so its
            # attention weight is exactly zero.
            kv_b = b_kv[start:end]
            z_b = np.full((m, a_z.shape[1]), -np.inf, dtype=float)
        else:
            prev_start = (i - 1) * m
            prev_end = i * m

            kv_b = b_kv[prev_start:prev_end]
            z_b = b_z[prev_start:prev_end] + B_b

        # Concatenate the 2m entries.
        kv = np.concatenate([kv_a, kv_b], axis=0)
        logits = np.concatenate([z_a, z_b], axis=0)

        # Stable softmax independently for each feature dimension.
        max_logits = np.max(logits, axis=0, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        weights = exp_logits / np.sum(exp_logits, axis=0, keepdims=True)

        # Weighted sum over the 2m tokens.
        compressed = np.sum(weights * kv, axis=0)
        outputs.append(compressed)

    return np.asarray(outputs).tolist()