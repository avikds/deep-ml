import numpy as np
import time
from sklearn.cluster import MiniBatchKMeans

def train(X_train):
    """Fit a Bernoulli mixture model using EM."""
    start = time.process_time()

    X = np.asarray(X_train, dtype=np.float32)
    n, d = X.shape

    K = 32 if n <= 6000 else 48
    floor = 0.002

    # Initialize mixture components with minibatch K-means.
    km = MiniBatchKMeans(
        n_clusters=K,
        batch_size=512,
        max_iter=45,
        n_init=1,
        random_state=42,
        reassignment_ratio=0.01
    )

    labels = km.fit_predict(X)

    counts = np.bincount(labels, minlength=K).astype(np.float32)

    pi = (counts + 0.1) / (n + 0.1 * K)

    mu = np.zeros((K, d), dtype=np.float32)

    for k in range(K):
        members = X[labels == k]

        if len(members):
            mu[k] = members.mean(axis=0)
        else:
            mu[k] = X.mean(axis=0)

    mu = np.clip(mu, floor, 1.0 - floor)

    # EM optimization.
    deadline = start + (53.0 if n <= 6000 else 108.0)
    previous_ll = -np.inf

    for iteration in range(80):
        # Bernoulli log-likelihood:
        # log p(x|k) = sum(log(1-mu)) + x @ log(mu/(1-mu))
        log_mu = np.log(mu)
        log_one_minus = np.log1p(-mu)

        log_odds = log_mu - log_one_minus
        base = log_one_minus.sum(axis=1)

        log_joint = X @ log_odds.T
        log_joint += base[None, :]
        log_joint += np.log(np.maximum(pi, 1e-12))[None, :]

        # Stable softmax for responsibilities.
        maximum = np.max(log_joint, axis=1, keepdims=True)
        exp_joint = np.exp(log_joint - maximum)
        totals = np.maximum(
            exp_joint.sum(axis=1, keepdims=True), 1e-30
        )

        resp = exp_joint / totals

        ll = np.mean(maximum[:, 0] + np.log(totals[:, 0]))

        # M-step.
        nk = resp.sum(axis=0)

        weighted = resp.T @ X

        mu_new = (
            weighted + 0.1
        ) / (
            nk[:, None] + 0.2
        )

        mu_new = np.clip(mu_new, floor, 1.0 - floor)

        pi_new = (nk + 0.1) / (n + 0.1 * K)

        mu = mu_new
        pi = pi_new

        if iteration >= 5 and abs(ll - previous_ll) < 1e-5:
            break

        previous_ll = ll

        if time.process_time() >= deadline:
            break

    # Precompute the likelihood contributions used at inference.
    log_mu = np.log(mu)
    log_one_minus = np.log1p(-mu)

    return {
        "pi": pi.astype(np.float32),
        "mu": mu.astype(np.float32),
        "log_mu": log_mu.astype(np.float32),
        "log_one_minus": log_one_minus.astype(np.float32),
        "log_odds": (log_mu - log_one_minus).astype(np.float32),
        "base": log_one_minus.sum(axis=1).astype(np.float32),
        "_logw": None,
        "_next_d": -1,
        "_n": -1
    }


def cond_prob(model, X_prefix):
    """
    Calculate P(pixel d = 1 | pixels 0,...,d-1).

    The grader calls this function in increasing order of d,
    so posterior component weights are updated incrementally.
    """
    X_prefix = np.asarray(X_prefix, dtype=np.float32)

    n, d = X_prefix.shape
    K = len(model["pi"])

    if d >= 784:
        raise ValueError("Prefix must contain fewer than 784 pixels")

    # At the first pixel, there is no evidence yet.
    if d == 0:
        logw = np.broadcast_to(
            np.log(model["pi"])[None, :],
            (n, K)
        ).copy()

    elif model["_next_d"] == d and model["_n"] == n:
        # Incorporate only the newly revealed pixel d-1.
        previous_pixel = X_prefix[:, d - 1:d]

        contribution = (
            previous_pixel * model["log_mu"][:, d - 1][None, :]
            + (1.0 - previous_pixel)
            * model["log_one_minus"][:, d - 1][None, :]
        )

        logw = model["_logw"] + contribution

    else:
        # Fallback for non-sequential calls.
        logw = (
            np.log(model["pi"])[None, :]
            + X_prefix @ model["log_odds"][:, :d].T
            + model["base"][None, :]
        )

    # Normalize log posterior weights.
    logw -= np.max(logw, axis=1, keepdims=True)

    weights = np.exp(logw)
    weights /= np.maximum(
        weights.sum(axis=1, keepdims=True), 1e-30
    )

    # Mixture conditional probability for the next pixel.
    p = weights @ model["mu"][:, d]

    # Retain posterior state for the next call.
    model["_logw"] = logw
    model["_next_d"] = d + 1
    model["_n"] = n

    return np.clip(p, 1e-6, 1.0 - 1e-6)