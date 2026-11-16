import numpy as np


def poly_fit(x, r, degree):
    """Fit a polynomial by least squares, with constant term first."""
    x = np.asarray(x, dtype=float)
    r = np.asarray(r, dtype=float)

    V = np.vander(x, N=degree + 1, increasing=True)
    coef = np.linalg.lstsq(V, r, rcond=None)[0]
    return coef


def poly_predict(x, coef):
    """Evaluate a polynomial with coefficients in increasing order."""
    x = np.asarray(x, dtype=float)
    coef = np.asarray(coef, dtype=float)

    return np.polynomial.polynomial.polyval(x, coef)


def backfit_gam(X, y, degree, n_iter=20):
    """Fit a generalized additive model using backfitting."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    n, p = X.shape
    alpha = float(np.mean(y))
    coefs = [None] * p
    F = np.zeros((n, p), dtype=float)

    for _ in range(n_iter):
        for j in range(p):
            # Partial residual excluding the current component.
            r = y - alpha - (F.sum(axis=1) - F[:, j])

            # Fit the polynomial smoother.
            coef = poly_fit(X[:, j], r, degree)
            fitted = poly_predict(X[:, j], coef)

            # Center the component and preserve its prediction.
            mean_fitted = np.mean(fitted)
            fitted -= mean_fitted
            coef[0] -= mean_fitted

            F[:, j] = fitted
            coefs[j] = coef

    return alpha, coefs, F


def gam_predict(X_new, alpha, coefs):
    """Predict using the fitted additive polynomial components."""
    X_new = np.asarray(X_new, dtype=float)

    predictions = np.full(X_new.shape[0], alpha, dtype=float)

    for j, coef in enumerate(coefs):
        predictions += poly_predict(X_new[:, j], coef)

    return predictions