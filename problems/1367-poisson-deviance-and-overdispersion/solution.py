import numpy as np


def poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    """Poisson deviance, using the convention 0 * log(0) = 0."""
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)

    # Avoid evaluating log(0) for y == 0.
    safe_y = np.where(y > 0, y, 1.0)

    terms = np.where(
        y > 0,
        y * np.log(safe_y / mu) - (y - mu),
        mu
    )

    return float(2.0 * np.sum(terms))


def dispersion_ratio(y: np.ndarray, mu: np.ndarray, n_params: int) -> float:
    """Pearson chi-square divided by (n - n_params)."""
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)

    n = len(y)
    pearson = np.sum((y - mu) ** 2 / mu)

    return float(pearson / (n - n_params))