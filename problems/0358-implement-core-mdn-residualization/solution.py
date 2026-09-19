import numpy as np

def mdn_layer(f: np.ndarray, X: np.ndarray, sigma_inv: np.ndarray, N: int) -> np.ndarray:
    """
    Apply Metadata Normalization (MDN) residualization.

    Uses the batch approximation from the MDN formulation:
        R = f - (N / M) * X @ sigma_inv @ X.T @ f

    where M is the current batch size and N is the full training-set size.
    """
    f = np.asarray(f, dtype=float)
    X = np.asarray(X, dtype=float)
    sigma_inv = np.asarray(sigma_inv, dtype=float)

    if f.ndim != 2 or X.ndim != 2 or sigma_inv.ndim != 2:
        raise ValueError("f, X, and sigma_inv must be 2D arrays")

    M, D = f.shape
    if X.shape[0] != M:
        raise ValueError("f and X must have the same number of samples")

    K = X.shape[1]
    if sigma_inv.shape != (K, K):
        raise ValueError("sigma_inv shape must be (K, K)")

    if M == 0:
        return f.copy()

    beta = (N / M) * (sigma_inv @ (X.T @ f))
    residual = f - X @ beta

    return residual