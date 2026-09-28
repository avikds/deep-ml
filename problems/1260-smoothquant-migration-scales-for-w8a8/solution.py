import numpy as np

def smoothquant(X: np.ndarray, W: np.ndarray, alpha: float = 0.5):
    """
    s_j = max|X_j|^alpha / max|W_j|^(1-alpha)
    X_smooth = X/s, W_smooth = W*s

    If denom 0 use 1e-8; if numerator 0 set s_j=1e-8.
    """
    x_max = np.max(np.abs(X), axis=0)
    w_max = np.max(np.abs(W), axis=0)

    # Avoid zero denominator.
    w_denom = np.maximum(w_max, 1e-8)

    s = (x_max ** alpha) / (w_denom ** (1.0 - alpha))

    # Explicit rule for zero numerator.
    s = np.where(x_max == 0, 1e-8, s)

    X_smooth = X / s
    W_smooth = W * s

    return s, X_smooth, W_smooth