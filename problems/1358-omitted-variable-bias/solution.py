import numpy as np


def ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least squares with an intercept."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    design = np.column_stack([np.ones(len(X)), X])
    coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    return coef


def omitted_variable_bias(X: np.ndarray, y: np.ndarray, omit_idx: int) -> tuple:
    """Return (full_kept, short, bias) for a two-column X."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    # Full two-regressor regression.
    full_coef = ols(X, y)

    kept_idx = 1 - omit_idx

    # Coefficient on the kept regressor in the full model.
    full_kept = full_coef[1 + kept_idx]

    # Regression with the omitted regressor removed.
    X_short = X[:, [kept_idx]]
    short_coef = ols(X_short, y)
    short = short_coef[1]

    # Auxiliary regression: omitted variable on kept variable.
    omitted = X[:, omit_idx]
    kept = X[:, kept_idx]
    aux_coef = ols(kept.reshape(-1, 1), omitted)
    delta = aux_coef[1]

    # Omitted-variable bias.
    beta_omitted = full_coef[1 + omit_idx]
    bias = beta_omitted * delta

    return float(full_kept), float(short), float(bias)