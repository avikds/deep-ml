import numpy as np
from scipy.stats import beta


def interval(sample, estimand, level):
    x = np.asarray(sample, dtype=np.float64)
    n = len(x)
    alpha = 1.0 - level

    if estimand == "proportion":
        s = int(np.sum(x))
        return (
            float(beta.ppf(alpha / 2, s + 0.5, n - s + 0.5)),
            float(beta.ppf(1 - alpha / 2, s + 0.5, n - s + 0.5))
        )

    rng = np.random.default_rng(0)
    B = 2048
    indices = rng.integers(0, n, size=(B, n))
    samples = x[indices]

    if estimand == "mean":
        estimates = np.mean(samples, axis=1)
        theta = float(np.mean(x))

    elif estimand == "median":
        estimates = np.median(samples, axis=1)
        theta = float(np.median(x))

    elif estimand == "variance":
        estimates = np.var(samples, axis=1, ddof=1)
        theta = float(np.var(x, ddof=1))

    else:
        raise ValueError("Unknown estimand")

    lo, hi = np.quantile(
        estimates, [alpha / 2, 1 - alpha / 2]
    )

    lo = min(float(lo), theta)
    hi = max(float(hi), theta)

    return lo, hi