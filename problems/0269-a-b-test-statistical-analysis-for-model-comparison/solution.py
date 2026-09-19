import numpy as np
import math

def analyze_ab_test(
    control_outcomes: list,
    treatment_outcomes: list,
    confidence_level: float = 0.95,
    min_detectable_effect: float = 0.02
) -> dict:

    if not control_outcomes or not treatment_outcomes:
        return {}

    control = np.asarray(control_outcomes, dtype=float)
    treatment = np.asarray(treatment_outcomes, dtype=float)

    n1 = len(control)
    n2 = len(treatment)

    p1 = np.mean(control)
    p2 = np.mean(treatment)

    absolute_lift = p2 - p1

    relative_lift_pct = (
        0.0 if p1 == 0 and p2 == 0
        else float("inf") if p1 == 0
        else (absolute_lift / p1) * 100.0
    )

    # Pooled two-proportion z-test
    pooled = (np.sum(control) + np.sum(treatment)) / (n1 + n2)
    pooled_var = pooled * (1.0 - pooled) * (1.0 / n1 + 1.0 / n2)

    if pooled_var == 0:
        z_statistic = 0.0 if absolute_lift == 0 else (
            float("inf") if absolute_lift > 0 else -float("inf")
        )
        p_value = 1.0 if absolute_lift == 0 else 0.0
    else:
        z_statistic = absolute_lift / math.sqrt(pooled_var)
        p_value = math.erfc(abs(z_statistic) / math.sqrt(2.0))

    alpha = 1.0 - confidence_level
    statistically_significant = p_value < alpha

    # Unpooled confidence interval
    se = math.sqrt(
        p1 * (1.0 - p1) / n1 +
        p2 * (1.0 - p2) / n2
    )

    # Inverse normal CDF using Acklam approximation
    def norm_ppf(p):
        a = [
            -39.6968302866538, 220.946098424521,
            -275.928510446969, 138.357751867269,
            -30.6647980661472, 2.50662827745924
        ]
        b = [
            -54.4760987982241, 161.585836858041,
            -155.698979859887, 66.8013118877197,
            -13.2806815528857
        ]
        c = [
            -0.00778489400243029, -0.322396458041136,
            -2.40075827716184, -2.54973253934373,
            4.37466414146497, 2.93816398269878
        ]
        d = [
            0.00778469570904146, 0.32246712907004,
            2.445134137143, 3.75440866190742
        ]

        plow = 0.02425
        phigh = 1.0 - plow

        if p < plow:
            q = math.sqrt(-2.0 * math.log(p))
            return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q +
                     c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q +
                     d[3]) * q + 1.0)

        if p > phigh:
            q = math.sqrt(-2.0 * math.log(1.0 - p))
            return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q +
                      c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q +
                      d[3]) * q + 1.0)

        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r +
                 a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r +
                 b[2]) * r + b[3]) * r + b[4]) * r + 1.0)

    z_crit = norm_ppf(1.0 - alpha / 2.0)

    ci_low = absolute_lift - z_crit * se
    ci_high = absolute_lift + z_crit * se

    # Practical significance
    practically_significant = (
        absolute_lift >= min_detectable_effect
    )

    # Required sample size per group for 80% power
    # Use the standard two-proportion approximation.
    if min_detectable_effect <= 0:
        required_sample_size = 0
    else:
        p_alt = min(1.0, p1 + min_detectable_effect)
        p_bar = (p1 + p_alt) / 2.0
        z_beta = norm_ppf(0.8)

        required_sample_size = math.ceil(
            (
                z_crit * math.sqrt(2.0 * p_bar * (1.0 - p_bar))
                + z_beta * math.sqrt(
                    p1 * (1.0 - p1) +
                    p_alt * (1.0 - p_alt)
                )
            ) ** 2 / (min_detectable_effect ** 2)
        )

    # Recommendation follows the stated decision rules directly.
    if not statistically_significant:
        recommendation = "continue_testing"
    elif absolute_lift > 0 and practically_significant:
        recommendation = "launch_treatment"
    else:
        recommendation = "keep_control"

    return {
        "control_rate": round(float(p1), 4),
        "treatment_rate": round(float(p2), 4),
        "absolute_lift": round(float(absolute_lift), 4),
        "relative_lift_pct": round(float(relative_lift_pct), 4),
        "z_statistic": round(float(z_statistic), 4),
        "p_value": round(float(p_value), 6),
        "confidence_interval": [
            round(float(ci_low), 4),
            round(float(ci_high), 4)
        ],
        "statistically_significant": bool(statistically_significant),
        "practically_significant": bool(practically_significant),
        "required_sample_size": int(required_sample_size),
        "sample_size_sufficient": bool(
            n1 >= required_sample_size and n2 >= required_sample_size
        ),
        "recommendation": recommendation
    }