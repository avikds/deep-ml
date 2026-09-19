import numpy as np

def loss_function(preds: np.ndarray, target: np.ndarray,
                  reduction: str = "mean", **kwargs):
    """
    preds:     [N, C] softmax probabilities (rows sum to 1)
    target:    [N]    class indices (int64)
    reduction: "mean", "sum", or "none"

    Returns:
        loss, grad
    """
    preds = np.asarray(preds)
    target = np.asarray(target, dtype=np.int64)

    N = preds.shape[0]

    # Numerical stability: avoid log(0) and division by zero.
    eps = 1e-12
    p = np.clip(preds, eps, 1.0)

    # Probability assigned to the correct class for each sample.
    correct_probs = p[np.arange(N), target]

    # Per-sample cross-entropy.
    losses = -np.log(correct_probs)

    # Gradient w.r.t. probabilities.
    grad = np.zeros_like(preds)
    grad[np.arange(N), target] = -1.0 / correct_probs

    if reduction == "mean":
        loss = np.mean(losses)
        grad /= N

    elif reduction == "sum":
        loss = np.sum(losses)

    elif reduction == "none":
        loss = losses

    else:
        raise ValueError(
            f"Invalid reduction '{reduction}'. "
            "Expected 'mean', 'sum', or 'none'."
        )

    return loss, grad