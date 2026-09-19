import numpy as np

def multi_head_latent_attention(
    X: np.ndarray,
    W_dkv: np.ndarray,
    W_uk: np.ndarray,
    W_uv: np.ndarray,
    W_dq: np.ndarray,
    W_uq: np.ndarray,
    W_o: np.ndarray,
    n_heads: int
) -> tuple:
    """
    Perform Multi-Head Latent Attention (MLA).
    """
    X = np.asarray(X, dtype=float)

    seq_len, d_model = X.shape

    if d_model % n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads")

    head_dim = d_model // n_heads

    # 1. Compress KV into latent representation
    c_kv = X @ W_dkv

    # 2. Reconstruct keys and values
    K = c_kv @ W_uk
    V = c_kv @ W_uv

    # 3. Compress and reconstruct queries
    c_q = X @ W_dq
    Q = c_q @ W_uq

    # Reshape into heads: (n_heads, seq_len, head_dim)
    Q_heads = Q.reshape(seq_len, n_heads, head_dim).transpose(1, 0, 2)
    K_heads = K.reshape(seq_len, n_heads, head_dim).transpose(1, 0, 2)
    V_heads = V.reshape(seq_len, n_heads, head_dim).transpose(1, 0, 2)

    # 4. Multi-head scaled dot-product attention
    scores = np.matmul(
        Q_heads,
        K_heads.transpose(0, 2, 1)
    ) / np.sqrt(head_dim)

    # Numerically stable softmax
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores)
    attn_weights = exp_scores / np.sum(
        exp_scores,
        axis=-1,
        keepdims=True
    )

    attended = np.matmul(attn_weights, V_heads)

    # Concatenate heads back to (seq_len, d_model)
    attended = attended.transpose(1, 0, 2).reshape(
        seq_len, d_model
    )

    # 5. Output projection
    output = attended @ W_o

    return output, c_kv