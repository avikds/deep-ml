import numpy as np

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray):
    """
    Compute Query (Q), Key (K), and Value (V) matrices.
    """
    return np.dot(X, W_q), np.dot(X, W_k), np.dot(X, W_v)


def masked_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: np.ndarray
) -> np.ndarray:
    """
    Compute masked self-attention.

    Args:
        Q: Query matrix of shape (seq_len, d_model).
        K: Key matrix of shape (seq_len, d_model).
        V: Value matrix of shape (seq_len, d_model).
        mask: Attention mask of shape (seq_len, seq_len),
              containing 0 for allowed positions and -inf for blocked positions.

    Returns:
        Attention output of shape (seq_len, d_model).
    """
    d_k = Q.shape[-1]

    # Scaled dot-product attention scores
    scores = np.dot(Q, K.T) / np.sqrt(d_k)

    # Apply causal mask
    scores = scores + mask

    # Numerically stable row-wise softmax
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # Weighted sum of values
    return np.dot(attention_weights, V)