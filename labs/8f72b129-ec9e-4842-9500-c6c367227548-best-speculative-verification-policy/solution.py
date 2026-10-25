import numpy as np

D = 5


def _features(X, mean, std):
    z = (X.astype(np.float64) - mean) / std

    # Mild nonlinear features improve calibration without a large model.
    return np.concatenate([
        z,
        z[:, [0, 1, 2, 5]] ** 2,
        (z[:, 0:1] * z[:, 1:2]),
        (z[:, 0:1] * z[:, 2:3]),
    ], axis=1)


def _sigmoid(x):
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def fit(train, t0, kappa):
    X = np.asarray(train["feats"], dtype=np.float64)
    y = np.asarray(train["accept"], dtype=np.float64)

    models = []

    for j in range(D):
        xj = X[:, :, j, :].reshape(-1, X.shape[-1])
        yj = y[:, :, j].reshape(-1)

        mean = xj.mean(axis=0)
        std = xj.std(axis=0)
        std = np.maximum(std, 1e-6)

        phi = _features(xj, mean, std)
        design = np.column_stack([
            np.ones(len(phi), dtype=np.float64),
            phi
        ])

        # Regularized logistic regression using Newton updates.
        w = np.zeros(design.shape[1], dtype=np.float64)
        reg = 0.02
        penalty = np.eye(len(w), dtype=np.float64) * reg
        penalty[0, 0] = 0.0

        for _ in range(35):
            p = _sigmoid(design @ w)
            grad = design.T @ (p - yj) / len(yj) + penalty @ w
            variance = np.maximum(p * (1.0 - p), 1e-5)

            hessian = (
                design.T @ (design * variance[:, None]) / len(yj)
                + penalty
            )

            try:
                update = np.linalg.solve(hessian, grad)
            except np.linalg.LinAlgError:
                update = np.linalg.pinv(hessian) @ grad

            w -= update

            if np.max(np.abs(update)) < 1e-6:
                break

        models.append({
            "mean": mean,
            "std": std,
            "weights": w,
        })

    return {
        "models": models,
        "t0": float(t0),
        "kappa": float(kappa),
    }


def plan(policy, feats):
    X = np.asarray(feats, dtype=np.float64)
    batch = X.shape[0]

    # Predict per-position acceptance probabilities.
    probs = np.empty((batch, D), dtype=np.float64)

    for j, model in enumerate(policy["models"]):
        phi = _features(X[:, j, :], model["mean"], model["std"])
        design = np.column_stack([
            np.ones(batch, dtype=np.float64),
            phi
        ])

        probs[:, j] = _sigmoid(design @ model["weights"])

    # Expected marginal accepted drafts from verifying each additional
    # position: P(all draft tokens up to and including that position pass).
    marginal = np.cumprod(probs, axis=1)
    marginal = np.clip(marginal, 1e-8, 1.0)

    # Increasing the number verified by one costs kappa milliseconds.
    # Find the best total number of verified drafts, accounting for the
    # fixed target-token contribution from every sequence.
    flat = marginal.ravel()
    order = np.argsort(-flat, kind="stable")
    sorted_gains = flat[order]

    cumulative = np.concatenate([
        np.array([0.0]),
        np.cumsum(sorted_gains)
    ])

    totals = np.arange(len(cumulative), dtype=np.float64)
    denom = policy["t0"] + policy["kappa"] * (batch + totals)
    denom = np.maximum(denom, 1e-12)

    throughput = (batch + cumulative) / denom
    best_total = int(np.argmax(throughput))

    # Count selected marginal gains for each sequence.
    lengths = np.zeros(batch, dtype=np.int64)
    if best_total > 0:
        selected = order[:best_total]
        sequence_ids = selected // D
        lengths = np.bincount(sequence_ids, minlength=batch).astype(np.int64)

    return np.clip(lengths, 0, D)