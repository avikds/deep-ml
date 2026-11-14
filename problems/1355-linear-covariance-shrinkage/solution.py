import numpy as np


def shrink_covariance(returns: np.ndarray, shrinkage: float) -> np.ndarray:
    S = np.cov(returns, rowvar=False, ddof=1)
    p = S.shape[0]
    mu = np.trace(S) / p
    F = mu * np.eye(p)
    return (1.0 - shrinkage) * S + shrinkage * F


def min_eigenvalue(matrix: np.ndarray) -> float:
    return float(np.linalg.eigvalsh(matrix)[0])