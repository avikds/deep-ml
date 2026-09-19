import numpy as np
import math

def two_sample_t_test(
    sample1: list[float],
    sample2: list[float],
    alpha: float = 0.05
) -> dict:
    x1 = np.asarray(sample1, dtype=float)
    x2 = np.asarray(sample2, dtype=float)

    n1, n2 = len(x1), len(x2)

    mean1 = np.mean(x1)
    mean2 = np.mean(x2)

    var1 = np.var(x1, ddof=1)
    var2 = np.var(x2, ddof=1)

    se = math.sqrt(var1 / n1 + var2 / n2)

    t_statistic = (mean1 - mean2) / se

    # Welch-Satterthwaite degrees of freedom
    a = var1 / n1
    b = var2 / n2

    degrees_of_freedom = (a + b) ** 2 / (
        (a ** 2) / (n1 - 1) +
        (b ** 2) / (n2 - 1)
    )

    # Regularized incomplete beta function
    def betacf(a, b, x):
        max_iter = 200
        eps = 3e-12

        qab = a + b
        qap = a + 1.0
        qam = a - 1.0

        c = 1.0
        d = 1.0 - qab * x / qap
        d = 1e-30 if abs(d) < 1e-30 else d
        d = 1.0 / d

        h = d

        for m in range(1, max_iter + 1):
            m2 = 2 * m

            aa = (
                m * (b - m) * x /
                ((qam + m2) * (a + m2))
            )

            d = 1.0 + aa * d
            d = 1e-30 if abs(d) < 1e-30 else d

            c = 1.0 + aa / c
            c = 1e-30 if abs(c) < 1e-30 else c

            d = 1.0 / d
            h *= d * c

            aa = -(
                (a + m) * (qab + m) * x /
                ((a + m2) * (qap + m2))
            )

            d = 1.0 + aa * d
            d = 1e-30 if abs(d) < 1e-30 else d

            c = 1.0 + aa / c
            c = 1e-30 if abs(c) < 1e-30 else c

            d = 1.0 / d

            delta = d * c
            h *= delta

            if abs(delta - 1.0) < eps:
                break

        return h

    def betainc(a, b, x):
        if x <= 0.0:
            return 0.0
        if x >= 1.0:
            return 1.0

        lbeta = (
            math.lgamma(a) +
            math.lgamma(b) -
            math.lgamma(a + b)
        )

        front = math.exp(
            a * math.log(x) +
            b * math.log(1.0 - x) -
            lbeta
        )

        if x < (a + 1.0) / (a + b + 2.0):
            return front * betacf(a, b, x) / a

        return 1.0 - front * betacf(b, a, 1.0 - x) / b

    # Two-tailed p-value
    x = degrees_of_freedom / (
        degrees_of_freedom + t_statistic ** 2
    )
    p_value = betainc(
        degrees_of_freedom / 2.0,
        0.5,
        x
    )

    # Cohen's d
    pooled_std = math.sqrt(
        (
            (n1 - 1) * var1 +
            (n2 - 1) * var2
        ) / (n1 + n2 - 2)
    )

    cohens_d = (mean1 - mean2) / pooled_std

    return {
        "t_statistic": round(float(t_statistic), 4),
        "p_value": round(float(p_value), 6),
        "degrees_of_freedom": round(float(degrees_of_freedom), 4),
        "reject_null": bool(p_value < alpha),
        "cohens_d": round(float(cohens_d), 4)
    }