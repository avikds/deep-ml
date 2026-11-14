import numpy as np


def combine_alphas(mu: np.ndarray, cov: np.ndarray, shrinkage: float) -> np.ndarray:
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)

    p = len(mu)
    avg_var = np.trace(cov) / p
    sigma = (1.0 - shrinkage) * cov + shrinkage * avg_var * np.eye(p)

    raw_weights = np.linalg.solve(sigma, mu)
    return raw_weights / raw_weights.sum()