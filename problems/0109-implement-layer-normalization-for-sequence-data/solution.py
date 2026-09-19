import numpy as np

def layer_normalization(
    X: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    epsilon: float = 1e-5
) -> np.ndarray:
    """
    Perform Layer Normalization on sequence data.

    Args:
        X: Input tensor of shape (batch_size, seq_len, d_model).
        gamma: Scale parameter, broadcastable to X.
        beta: Shift parameter, broadcastable to X.
        epsilon: Small constant for numerical stability.

    Returns:
        Layer-normalized tensor with the same shape as X.
    """
    # Compute statistics independently for each sequence position
    # across the feature dimension.
    mean = np.mean(X, axis=-1, keepdims=True)
    variance = np.var(X, axis=-1, keepdims=True)

    # Normalize
    X_normalized = (X - mean) / np.sqrt(variance + epsilon)

    # Apply learnable scale and shift
    return gamma * X_normalized + beta