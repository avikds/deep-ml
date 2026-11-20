import numpy as np


def _sigmoid(z):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def platt_fit(scores, y, n_iter=100):
    s = np.asarray(scores, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    X = np.column_stack((s, np.ones_like(s)))
    theta = np.array([1.0, 0.0])

    for _ in range(n_iter):
        z = X @ theta
        p = _sigmoid(z)

        grad = X.T @ (p - y) / len(y)
        w = p * (1.0 - p)

        H = (X.T * w) @ X / len(y)

        # Small damping protects against singular Hessians.
        H += np.eye(2) * 1e-12

        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            break

        if np.max(np.abs(step)) < 1e-10:
            break

        old_loss = np.mean(np.logaddexp(0.0, z) - y * z)
        scale = 1.0

        # Backtracking line search.
        while scale > 1e-8:
            candidate = theta - scale * step
            z_new = X @ candidate
            new_loss = np.mean(
                np.logaddexp(0.0, z_new) - y * z_new
            )
            if new_loss <= old_loss:
                break
            scale *= 0.5

        if scale <= 1e-8:
            break

        theta = candidate

        if np.max(np.abs(scale * step)) < 1e-10:
            break

    return float(theta[0]), float(theta[1])


def platt_predict(scores, a, b):
    return _sigmoid(a * np.asarray(scores, dtype=float) + b)


def isotonic_fit(scores, y):
    scores = np.asarray(scores, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    xs, inverse, counts = np.unique(
        scores, return_inverse=True, return_counts=True
    )

    sums = np.bincount(inverse, weights=y)
    means = sums / counts

    # PAV blocks: [start, end, total_weight, weighted_sum]
    blocks = []

    for i in range(len(xs)):
        blocks.append([i, i, float(counts[i]), float(sums[i])])

        while len(blocks) >= 2:
            left = blocks[-2]
            right = blocks[-1]

            left_mean = left[3] / left[2]
            right_mean = right[3] / right[2]

            if left_mean <= right_mean:
                break

            merged = [
                left[0],
                right[1],
                left[2] + right[2],
                left[3] + right[3]
            ]

            blocks[-2:] = [merged]

    fitted = np.empty(len(xs), dtype=float)

    for start, end, weight, total in blocks:
        fitted[start:end + 1] = total / weight

    return xs, fitted


def isotonic_predict(scores, xs, ys):
    scores = np.asarray(scores, dtype=float)
    return np.interp(scores, xs, ys)


def expected_calibration_error(p, y, n_bins=10):
    p = np.asarray(p, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    if len(p) == 0:
        return 0.0

    p = np.clip(p, 0.0, 1.0)
    bin_ids = np.minimum((p * n_bins).astype(int), n_bins - 1)

    ece = 0.0

    for b in range(n_bins):
        mask = bin_ids == b
        if np.any(mask):
            ece += (
                np.sum(mask) / len(p)
                * abs(np.mean(p[mask]) - np.mean(y[mask]))
            )

    return float(ece)


def calibration_report(scores, y_fit, scores_test, y_test):
    a, b = platt_fit(scores, y_fit)

    xs, ys = isotonic_fit(scores, y_fit)

    raw = np.asarray(scores_test, dtype=float)
    platt = platt_predict(scores_test, a, b)
    isotonic = isotonic_predict(scores_test, xs, ys)

    return {
        "ece_raw": round(expected_calibration_error(raw, y_test), 4),
        "ece_platt": round(expected_calibration_error(platt, y_test), 4),
        "ece_isotonic": round(expected_calibration_error(isotonic, y_test), 4)
    }