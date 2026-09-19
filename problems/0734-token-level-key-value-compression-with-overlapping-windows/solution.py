import numpy as np

def compressed_kv_entries(H, W_aKV, W_bKV, W_aZ, W_bZ, B_a, B_b, m, stride):
    """
    H: (n, d) hidden states
    W_aKV, W_bKV: (d, c) KV projection weights for the two streams
    W_aZ, W_bZ: (d, c) compression-weight projection matrices
    B_a, B_b: (m, c) learnable positional biases
    m: window size
    stride: step between consecutive windows

    Returns:
        list of lists of compressed KV entries, shape
        ((n - m) // stride + 1, c)
    """
    H = np.asarray(H, dtype=float)
    W_aKV = np.asarray(W_aKV, dtype=float)
    W_bKV = np.asarray(W_bKV, dtype=float)
    W_aZ = np.asarray(W_aZ, dtype=float)
    W_bZ = np.asarray(W_bZ, dtype=float)
    B_a = np.asarray(B_a, dtype=float)
    B_b = np.asarray(B_b, dtype=float)

    n = H.shape[0]
    num_groups = (n - m) // stride + 1

    # Project the full sequence once.
    a_kv = H @ W_aKV
    b_kv = H @ W_bKV
    a_z = H @ W_aZ
    b_z = H @ W_bZ

    outputs = []

    for g in range(num_groups):
        start = g * stride
        end = start + m

        # Windowed KV entries: [a_0 ... a_m-1, b_0 ... b_m-1]
        kv = np.concatenate(
            [a_kv[start:end], b_kv[start:end]],
            axis=0
        )

        # Compression logits with positional biases.
        z = np.concatenate(
            [
                a_z[start:end] + B_a,
                b_z[start:end] + B_b
            ],
            axis=0
        )

        # Softmax across the 2*m tokens, independently for each feature.
        z = z - np.max(z, axis=0, keepdims=True)
        weights = np.exp(z)
        weights /= np.sum(weights, axis=0, keepdims=True)

        # Weighted sum of KV entries.
        compressed = np.sum(weights * kv, axis=0)

        outputs.append(compressed)

    return np.asarray(outputs).tolist()