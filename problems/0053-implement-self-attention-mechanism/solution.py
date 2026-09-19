import numpy as np

def compute_qkv(X, W_q, W_k, W_v):
    """Compute Query, Key, Value matrices from input X and weight matrices."""
    Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)
    return Q, K, V


def self_attention(Q, K, V):
    """
    Compute scaled dot-product self-attention.

    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)

    Returns:
        Attention output of shape (seq_len, d_v)
    """
    d_k = K.shape[1]

    # Step 1: Scaled dot-product attention scores
    scores = np.dot(Q, K.T) / np.sqrt(d_k)

    # Step 2: Numerically stable row-wise softmax
    scores_max = np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attention_weights = exp_scores / np.sum(
        exp_scores, axis=1, keepdims=True
    )

    # Step 3: Weighted sum of values
    output = np.dot(attention_weights, V)

    return output