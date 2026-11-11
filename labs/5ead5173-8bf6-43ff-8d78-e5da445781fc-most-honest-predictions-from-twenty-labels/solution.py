import numpy as np
from scipy.special import expit, logsumexp


def fit_predict(X_train, y_train, X_test):
    X = np.asarray(X_train, dtype=np.float64)
    T = np.asarray(X_test, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)

    n, d = X.shape

    # ------------------------------------------------
    # 1. Standardize using training statistics
    # ------------------------------------------------

    mean = X.mean(axis=0)
    std = X.std(axis=0)

    std = np.maximum(std, 1e-8)

    X = (X - mean) / std
    T = (T - mean) / std

    # Add intercept.
    A = np.column_stack([np.ones(n), X])
    B = np.column_stack([np.ones(len(T)), T])

    p = d + 1

    # ------------------------------------------------
    # 2. Bayesian prior
    # ------------------------------------------------

    # Intercept has a relatively broad fixed prior.
    intercept_precision = 1.0 / 9.0

    # Integrate over unknown weight scales.
    scales = np.geomspace(0.08, 8.0, 17)

    log_evidences = []
    posterior_models = []

    # ------------------------------------------------
    # 3. Fit each prior scale using Newton optimization
    # ------------------------------------------------

    initial = np.zeros(p)

    freq = np.clip(y.mean(), 1e-5, 1 - 1e-5)
    initial[0] = np.log(freq / (1 - freq))

    for scale in scales:

        precision = np.full(p, 1.0 / (scale * scale))
        precision[0] = intercept_precision

        prior = np.diag(precision)

        theta = initial.copy()

        # MAP optimization.
        for iteration in range(60):

            eta = A @ theta
            prob = expit(eta)

            gradient = A.T @ (prob - y) + prior @ theta

            variance = prob * (1.0 - prob)

            H = A.T @ (A * variance[:, None]) + prior

            try:
                step = np.linalg.solve(H, gradient)
            except np.linalg.LinAlgError:
                break

            # Newton step with backtracking.
            old_loss = (
                np.logaddexp(0.0, eta).sum()
                - np.dot(y, eta)
                + 0.5 * np.dot(theta, prior @ theta)
            )

            step_size = 1.0

            while step_size > 1e-7:

                candidate = theta - step_size * step
                candidate_eta = A @ candidate

                new_loss = (
                    np.logaddexp(0.0, candidate_eta).sum()
                    - np.dot(y, candidate_eta)
                    + 0.5 * np.dot(candidate, prior @ candidate)
                )

                if new_loss <= old_loss:
                    break

                step_size *= 0.5

            theta = candidate

            if np.max(np.abs(step_size * step)) < 1e-7:
                break

        # ------------------------------------------------
        # 4. Laplace approximation
        # ------------------------------------------------

        eta = A @ theta
        prob = expit(eta)
        variance = prob * (1.0 - prob)

        H = A.T @ (A * variance[:, None]) + prior

        try:
            L = np.linalg.cholesky(H)

            # log determinant of Hessian.
            logdet_H = 2.0 * np.log(np.diag(L)).sum()

            # Posterior covariance.
            covariance = np.linalg.solve(
                L.T,
                np.linalg.solve(L, np.eye(p))
            )

        except np.linalg.LinAlgError:
            continue

        # Laplace log marginal likelihood.
        log_likelihood = np.sum(
            y * eta - np.logaddexp(0.0, eta)
        )

        log_prior_kernel = -0.5 * np.dot(
            theta, prior @ theta
        )

        logdet_prior = np.log(precision).sum()

        log_evidence = (
            log_likelihood
            + log_prior_kernel
            + 0.5 * logdet_prior
            - 0.5 * logdet_H
        )

        log_evidences.append(log_evidence)

        posterior_models.append((theta, covariance))

        initial = theta.copy()

    # ------------------------------------------------
    # 5. Bayesian model averaging
    # ------------------------------------------------

    if len(posterior_models) == 0:
        return np.full(len(T), freq, dtype=np.float64)

    log_evidences = np.asarray(log_evidences)

    model_weights = np.exp(
        log_evidences - logsumexp(log_evidences)
    )

    # ------------------------------------------------
    # 6. Posterior predictive probabilities
    # ------------------------------------------------

    predictions = np.zeros(len(T))

    for weight, (theta, covariance) in zip(
        model_weights, posterior_models
    ):

        mu = B @ theta

        # Predictive variance from parameter uncertainty.
        projected = B @ covariance

        var = np.sum(projected * B, axis=1)

        var = np.maximum(var, 0.0)

        # Logistic-normal approximation:
        # E[sigmoid(Z)] where Z ~ N(mu, var).
        adjusted_mu = mu / np.sqrt(
            1.0 + np.pi * var / 8.0
        )

        predictions += weight * expit(adjusted_mu)

    # Keep all predictions strictly inside (0, 1).
    return np.clip(predictions, 1e-6, 1.0 - 1e-6)