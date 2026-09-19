import numpy as np

def overlapping_kv_compression(Z_a, Z_b, B_a, B_b, C_a, C_b):
    """
    Compute compressed KV entries from two overlapping score/value blocks.

    Returns:
        Nested list of shape (N, D).
    """
    Z_a = np.asarray(Z_a, dtype=float)
    Z_b = np.asarray(Z_b, dtype=float)
    B_a = np.asarray(B_a, dtype=float)
    B_b = np.asarray(B_b, dtype=float)
    C_a = np.asarray(C_a, dtype=float)
    C_b = np.asarray(C_b, dtype=float)

    # Add positional biases.
    biased_a = Z_a + B_a
    biased_b = Z_b + B_b

    # Joint scores over both blocks.
    scores = np.concatenate([biased_a, biased_b], axis=1)

    # Numerically stable softmax.
    scores = scores - np.max(scores, axis=1, keepdims=True)
    weights = np.exp(scores)
    weights /= np.sum(weights, axis=1, keepdims=True)

    # Split joint attention weights back into A and B.
    m_a = Z_a.shape[1]
    W_a = weights[:, :m_a]
    W_b = weights[:, m_a:]

    # Jointly compressed values.
    C_comp = W_a @ C_a + W_b @ C_b

    return C_comp.tolist()