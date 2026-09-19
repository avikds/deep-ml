import math

def compare_softmax(scores: list) -> dict:
    """Compare naive and numerically stable softmax."""

    # Naive softmax
    try:
        exp_scores = [math.exp(x) for x in scores]
        total = sum(exp_scores)
        naive = [x / total for x in exp_scores]
    except OverflowError:
        # Match the expected unstable behavior for overflow.
        exp_scores = []
        for x in scores:
            try:
                exp_scores.append(math.exp(x))
            except OverflowError:
                exp_scores.append(float("inf"))

        total = sum(exp_scores)
        naive = [
            x / total if math.isfinite(x) and math.isfinite(total)
            else float("nan")
            for x in exp_scores
        ]

    # Numerically stable softmax
    max_score = max(scores)
    exp_stable = [math.exp(x - max_score) for x in scores]
    total_stable = sum(exp_stable)
    stable = [x / total_stable for x in exp_stable]

    # Treat any NaN in naive/stable as a mismatch.
    if any(math.isnan(x) for x in naive) or any(math.isnan(x) for x in stable):
        max_abs_diff = float("nan")
    else:
        max_abs_diff = max(
            abs(a - b) for a, b in zip(naive, stable)
        )

    return {
        "naive": [
            float("nan") if math.isnan(x) else round(x, 6)
            for x in naive
        ],
        "stable": [round(x, 6) for x in stable],
        "max_abs_diff": (
            float("nan")
            if math.isnan(max_abs_diff)
            else round(max_abs_diff, 6)
        )
    }