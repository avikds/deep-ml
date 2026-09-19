import numpy as np
from typing import Tuple


def compute_qkv(
    X: np.ndarray,
    W_q: np.ndarray,
    W_k: np.ndarray,
    W_v: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Query, Key, and Value matrices.

    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)

    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """
    Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)

    return Q, K, V


def self_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray
) -> np.ndarray:
    """
    Compute scaled dot-product self-attention.

    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)

    Returns:
        Attention output of shape (seq_len, d_k)
    """
    d_k = Q.shape[-1]

    # Compute scaled attention scores
    scores = np.dot(Q, K.T) / np.sqrt(d_k)

    # Numerically stable softmax
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attention_weights = exp_scores / np.sum(
        exp_scores, axis=-1, keepdims=True
    )

    # Weighted sum of values
    return np.dot(attention_weights, V)


def multi_head_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    n_heads: int
) -> np.ndarray:
    """
    Compute multi-head attention.

    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads

    Returns:
        Attention output of shape (seq_len, d_model)
    """
    seq_len, d_model = Q.shape

    # Each head gets an equal-sized slice of the feature dimension
    assert d_model % n_heads == 0, \
        "d_model must be divisible by n_heads"

    d_head = d_model // n_heads

    head_outputs = []

    for i in range(n_heads):
        start = i * d_head
        end = (i + 1) * d_head

        Q_head = Q[:, start:end]
        K_head = K[:, start:end]
        V_head = V[:, start:end]

        head_output = self_attention(Q_head, K_head, V_head)
        head_outputs.append(head_output)

    # Concatenate all heads along the feature dimension
    return np.concatenate(head_outputs, axis=-1)