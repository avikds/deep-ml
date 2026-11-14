import math
import numpy as np


def ols_inference(X: np.ndarray, y: np.ndarray) -> tuple:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    n, p = X.shape
    D = np.column_stack((np.ones(n), X))

    # Least-squares coefficients
    coef = np.linalg.lstsq(D, y, rcond=None)[0]

    # Residual variance
    residuals = y - D @ coef
    rss = float(residuals @ residuals)
    sigma2 = rss / (n - p - 1)

    # Covariance matrix and standard errors
    cov_beta = sigma2 * np.linalg.inv(D.T @ D)
    se = np.sqrt(np.maximum(np.diag(cov_beta), 0.0))

    # t-statistics
    t = np.divide(
        coef, se,
        out=np.zeros_like(coef),
        where=se != 0
    )

    # Two-sided p-values using the normal approximation
    p_values = np.array([
        math.erfc(abs(float(value)) / math.sqrt(2.0))
        for value in t
    ])

    return coef, se, t, p_values