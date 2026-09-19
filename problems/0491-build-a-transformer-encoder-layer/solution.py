import numpy as np

def transformer_encoder_layer(
    X: np.ndarray,
    weights: dict,
    num_heads: int,
    eps: float = 1e-5
) -> np.ndarray:
    """
    Forward pass of a single Transformer Encoder Layer.

    Args:
        X: Input tensor of shape (batch_size, seq_len, d_model)
        weights: Dictionary containing all weight matrices and normalization parameters
        num_heads: Number of attention heads
        eps: Epsilon for layer normalization

    Returns:
        Output tensor of shape (batch_size, seq_len, d_model)
    """

    batch_size, seq_len, d_model = X.shape

    if d_model % num_heads != 0:
        raise ValueError("d_model must be evenly divisible by num_heads")

    head_dim = d_model // num_heads

    # ---------------------------------------------------------
    # 1. Multi-Head Self-Attention
    # ---------------------------------------------------------

    # Project to queries, keys, and values.
    Q = X @ weights["W_q"]
    K = X @ weights["W_k"]
    V = X @ weights["W_v"]

    # Split into heads:
    # (B, S, D) -> (B, H, S, D_head)
    Q = Q.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    K = K.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    V = V.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)

    # Scaled dot-product attention.
    scores = np.matmul(Q, K.transpose(0, 1, 3, 2))
    scores = scores / np.sqrt(head_dim)

    # Numerically stable softmax over the key/sequence dimension.
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    attention_weights = np.exp(scores)
    attention_weights /= np.sum(
        attention_weights,
        axis=-1,
        keepdims=True
    )

    # Weighted sum of values.
    context = np.matmul(attention_weights, V)

    # Combine heads:
    # (B, H, S, D_head) -> (B, S, D)
    context = context.transpose(0, 2, 1, 3)
    context = context.reshape(batch_size, seq_len, d_model)

    # Output projection.
    attention_output = context @ weights["W_o"]

    # ---------------------------------------------------------
    # 2. Add & Norm
    # ---------------------------------------------------------

    residual1 = X + attention_output

    mean1 = np.mean(residual1, axis=-1, keepdims=True)
    var1 = np.mean(
        (residual1 - mean1) ** 2,
        axis=-1,
        keepdims=True
    )

    norm1 = (residual1 - mean1) / np.sqrt(var1 + eps)
    norm1 = (
        norm1 * weights["gamma1"]
        + weights["beta1"]
    )

    # ---------------------------------------------------------
    # 3. Position-wise Feed-Forward Network
    # ---------------------------------------------------------

    hidden = norm1 @ weights["W1"] + weights["b1"]

    # ReLU
    hidden = np.maximum(0.0, hidden)

    ffn_output = hidden @ weights["W2"] + weights["b2"]

    # ---------------------------------------------------------
    # 4. Add & Norm
    # ---------------------------------------------------------

    residual2 = norm1 + ffn_output

    mean2 = np.mean(residual2, axis=-1, keepdims=True)
    var2 = np.mean(
        (residual2 - mean2) ** 2,
        axis=-1,
        keepdims=True
    )

    norm2 = (residual2 - mean2) / np.sqrt(var2 + eps)
    norm2 = (
        norm2 * weights["gamma2"]
        + weights["beta2"]
    )

    return norm2