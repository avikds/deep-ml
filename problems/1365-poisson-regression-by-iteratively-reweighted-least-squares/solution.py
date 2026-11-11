import numpy as np

def fit_poisson_irls(X: np.ndarray, y: np.ndarray, n_iter: int) -> np.ndarray:
    """Fit log(lambda) = D @ beta by iteratively reweighted least squares."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    n = X.shape[0]
    D = np.column_stack((np.ones(n), X))
    beta = np.zeros(D.shape[1], dtype=float)

    for _ in range(n_iter):
        eta = D @ beta
        mu = np.exp(np.clip(eta, -700, 700))

        # Poisson IRLS working response and weights
        z = eta + (y - mu) / mu
        w = mu

        # Solve (D.T @ W @ D) beta = D.T @ W @ z
        lhs = D.T @ (w[:, None] * D)
        rhs = D.T @ (w * z)

        beta = np.linalg.solve(lhs, rhs)

    return beta