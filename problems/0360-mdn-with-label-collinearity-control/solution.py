import numpy as np

def mdn_with_collinearity(
    f: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    sigma_tilde_inv: np.ndarray,
    N: int
) -> np.ndarray:
    """
    Remove metadata-related variation from features while preserving
    variation explained by the label.
    """
    f = np.asarray(f, dtype=float)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if f.ndim != 2 or X.ndim != 2:
        raise ValueError("f and X must be 2D")

    if y.ndim == 1:
        y = y[:, None]
    elif y.ndim != 2 or y.shape[1] != 1:
        raise ValueError("y must have shape (M,) or (M, 1)")

    M = f.shape[0]

    if X.shape[0] != M or y.shape[0] != M:
        raise ValueError("f, X, and y must have the same number of samples")

    K = X.shape[1]

    if sigma_tilde_inv.shape != (K + 1, K + 1):
        raise ValueError("sigma_tilde_inv must have shape (K+1, K+1)")

    if M == 0:
        return f.copy()

    # Augmented metadata + label matrix.
    Z = np.concatenate([X, y], axis=1)

    # Joint regression coefficients for features on [X, y].
    # The N/M factor matches the batch MDN scaling.
    coef = (N / M) * (sigma_tilde_inv @ (Z.T @ f))

    # Only remove the metadata contribution.
    # Preserve the part explained by the label.
    metadata_component = X @ coef[:K, :]

    return f - metadata_component