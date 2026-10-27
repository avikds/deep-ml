import numpy as np
import math


def logit_margin(z, t):
    """Margin m = z_t - log sum_{j != t} exp(z_j), computed stably."""
    z = np.asarray(z, dtype=np.float64)
    other = np.concatenate((z[:t], z[t + 1:]))

    max_other = np.max(other)
    logsumexp = max_other + np.log(np.sum(np.exp(other - max_other)))

    return float(z[t] - logsumexp)


def precision_cliff(dtype):
    """Find the smallest margin where the sigmoid rounds to exactly 1."""
    dt = np.dtype(dtype).type
    one = dt(1.0)

    def saturated(m):
        # Cast e^(-m) to the requested dtype before adding 1.
        exp_term = dt(math.exp(-m))
        denominator = dt(one + exp_term)
        sigmoid = dt(one / denominator)
        return sigmoid == one

    lo, hi = 0.0, 100.0

    # The saturation predicate is monotone in m.
    while hi - lo >= 1e-9:
        mid = (lo + hi) / 2.0
        if saturated(mid):
            hi = mid
        else:
            lo = mid

    return round(hi, 4)