import numpy as np

def flash_attention_forward(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    block_size: int = 2
) -> np.ndarray:
    """
    Compute attention output using Flash Attention v1 algorithm.
    """
    Q = np.asarray(Q, dtype=float)
    K = np.asarray(K, dtype=float)
    V = np.asarray(V, dtype=float)

    n, d = Q.shape
    scale = 1.0 / np.sqrt(d)

    # Running max, denominator, and unnormalized output
    m = np.full(n, -np.inf, dtype=float)
    l = np.zeros(n, dtype=float)
    O = np.zeros((n, V.shape[1]), dtype=float)

    for start in range(0, K.shape[0], block_size):
        end = min(start + block_size, K.shape[0])

        K_block = K[start:end]
        V_block = V[start:end]

        # Attention scores for this tile
        S = (Q @ K_block.T) * scale

        # Maximum score in current block
        m_block = np.max(S, axis=1)

        # Stable block softmax
        P = np.exp(S - m_block[:, None])
        l_block = np.sum(P, axis=1)

        # Merge current block with running statistics
        m_new = np.maximum(m, m_block)

        old_scale = np.exp(m - m_new)
        block_scale = np.exp(m_block - m_new)

        l = old_scale * l + block_scale * l_block

        O = (
            old_scale[:, None] * O
            + block_scale[:, None] * (P @ V_block)
        )

        m = m_new

    # Normalize the accumulated numerator
    O /= np.maximum(l[:, None], 1e-12)

    return O