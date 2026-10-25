import numpy as np

WIDTH = 512
K = 32


def train(X_train):
    X = np.asarray(X_train, dtype=np.float32)
    mu = X.mean(axis=0, dtype=np.float64).astype(np.float32)
    centered = X - mu

    # PCA dictionary: rows of components are unit-norm directions.
    cov = (centered.T @ centered) / max(1, X.shape[0])
    cov = cov.astype(np.float64)
    cov = 0.5 * (cov + cov.T)

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    components = eigvecs[:, order].T.astype(np.float32)

    # Absolute projection scores determine which latents are selected.
    scores = np.abs(centered @ components.T)
    del centered, cov, eigvals, eigvecs

    # Calibrate biases so no latent dominates top-K selection.
    bias = np.zeros(WIDTH, dtype=np.float32)
    target_frequency = 0.38

    for _ in range(12):
        adjusted = scores + bias[None, :]

        indices = np.argpartition(
            adjusted, WIDTH - K, axis=1
        )[:, WIDTH - K:]

        counts = np.bincount(indices.ravel(), minlength=WIDTH)
        frequency = counts / X.shape[0]

        over = np.flatnonzero(frequency > target_frequency)
        if len(over) == 0:
            break

        # Per-example cutoff for the Kth selected latent.
        cutoff = np.partition(
            adjusted, WIDTH - K, axis=1
        )[:, WIDTH - K]

        # Lower priority for latents selected too frequently.
        margins = adjusted[:, over] - cutoff[:, None]
        correction = np.quantile(
            margins, 1.0 - target_frequency, axis=0
        )
        bias[over] -= np.maximum(correction, 0.0).astype(np.float32) + 1e-6

    return {
        "mu": mu,
        "components": components,
        "bias": bias,
    }


def encode(model, X):
    X = np.asarray(X, dtype=np.float32)
    scores = (X - model["mu"]) @ model["components"].T

    # Bias affects selection only, not the coefficient values.
    adjusted = np.abs(scores) + model["bias"][None, :]
    idx = np.argpartition(
        adjusted, WIDTH - K, axis=1
    )[:, WIDTH - K:]

    Z = np.zeros_like(scores, dtype=np.float32)
    np.put_along_axis(
        Z, idx, np.take_along_axis(scores, idx, axis=1), axis=1
    )
    return Z


def decode(model, Z):
    Z = np.asarray(Z, dtype=np.float32)
    return Z @ model["components"] + model["mu"]