import numpy as np
import math


def matern_kernel(x: np.ndarray, x_prime: np.ndarray, length_scale=1.0, nu=1.5):
    x = np.asarray(x, dtype=float)
    x_prime = np.asarray(x_prime, dtype=float)

    r = np.linalg.norm(x - x_prime)
    if r == 0:
        return 1.0

    z = np.sqrt(2.0 * nu) * r / length_scale

    if nu == 0.5:
        return np.exp(-z)

    if nu == 1.5:
        return (1.0 + z) * np.exp(-z)

    if nu == 2.5:
        return (1.0 + z + z**2 / 3.0) * np.exp(-z)

    # General Matern form
    from math import gamma
    from scipy.special import kv

    return (
        (2.0 ** (1.0 - nu)) / gamma(nu)
        * z**nu
        * kv(nu, z)
    )


def rbf_kernel(x: np.ndarray, x_prime, sigma=1.0, length_scale=1.0):
    x = np.asarray(x, dtype=float)
    x_prime = np.asarray(x_prime, dtype=float)

    sq_dist = np.sum((x - x_prime) ** 2)
    return sigma**2 * np.exp(-0.5 * sq_dist / (length_scale**2))


def periodic_kernel(
    x: np.ndarray,
    x_prime: np.ndarray,
    sigma=1.0,
    length_scale=1.0,
    period=1.0
):
    x = np.asarray(x, dtype=float)
    x_prime = np.asarray(x_prime, dtype=float)

    dist = np.linalg.norm(x - x_prime)
    return sigma**2 * np.exp(
        -2.0 * np.sin(np.pi * dist / period) ** 2
        / (length_scale**2)
    )


def linear_kernel(
    x: np.ndarray,
    x_prime: np.ndarray,
    sigma_b=1.0,
    sigma_v=1.0
):
    x = np.asarray(x, dtype=float)
    x_prime = np.asarray(x_prime, dtype=float)

    return sigma_b**2 + sigma_v**2 * np.dot(x, x_prime)


def rational_quadratic_kernel(
    x: np.ndarray,
    x_prime: np.ndarray,
    sigma=1.0,
    length_scale=1.0,
    alpha=1.0
):
    x = np.asarray(x, dtype=float)
    x_prime = np.asarray(x_prime, dtype=float)

    sq_dist = np.sum((x - x_prime) ** 2)

    return sigma**2 * (
        1.0 + sq_dist / (2.0 * alpha * length_scale**2)
    ) ** (-alpha)


# --- BASE CLASS -------------------------------------------------------------


class _GaussianProcessBase:
    def __init__(self, kernel="rbf", noise=1e-5, kernel_params=None):
        self.kernel = kernel
        self.noise = noise
        self.kernel_params = kernel_params or {}

        kernels = {
            "rbf": rbf_kernel,
            "matern": matern_kernel,
            "periodic": periodic_kernel,
            "linear": linear_kernel,
            "rational_quadratic": rational_quadratic_kernel,
        }

        if kernel not in kernels:
            raise ValueError(f"Unknown kernel: {kernel}")

        self.kernel_func = kernels[kernel]

    def _select_kernel(self, x1, x2):
        return self.kernel_func(
            np.asarray(x1),
            np.asarray(x2),
            **self.kernel_params
        )

    def _compute_covariance(self, X1, X2):
        X1 = np.asarray(X1, dtype=float)
        X2 = np.asarray(X2, dtype=float)

        return np.array([
            [self._select_kernel(x1, x2) for x2 in X2]
            for x1 in X1
        ])


# --- REGRESSION MODEL -------------------------------------------------------


class GaussianProcessRegression(_GaussianProcessBase):
    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y, dtype=float).reshape(-1)

        K = self._compute_covariance(self.X_train, self.X_train)

        # Observation noise
        K = K + self.noise * np.eye(len(self.X_train))

        # Small jitter for numerical stability
        K = K + 1e-10 * np.eye(len(self.X_train))

        self.K = K

        try:
            self.L = np.linalg.cholesky(K)
            tmp = np.linalg.solve(self.L, self.y_train)
            self.alpha = np.linalg.solve(self.L.T, tmp)
        except np.linalg.LinAlgError:
            self.K_inv = np.linalg.pinv(K)
            self.alpha = self.K_inv @ self.y_train

        return self

    def predict(self, X_test, return_std=False):
        X_test = np.asarray(X_test, dtype=float)

        K_star = self._compute_covariance(X_test, self.X_train)

        mu = K_star @ self.alpha

        if not return_std:
            return np.asarray(mu)

        K_ss = self._compute_covariance(X_test, X_test)

        try:
            v = np.linalg.solve(self.L, K_star.T)
            var = np.diag(K_ss) - np.sum(v**2, axis=0)
        except AttributeError:
            K_inv = np.linalg.pinv(self.K)
            var = np.diag(K_ss) - np.sum(
                (K_star @ K_inv) * K_star,
                axis=1
            )

        var = np.maximum(var, 0.0)

        # Predictive standard deviation including observation noise
        std = np.sqrt(var + self.noise)

        return np.asarray(mu), std

    def log_marginal_likelihood(self):
        if not hasattr(self, "X_train"):
            raise ValueError("Model must be fitted before computing likelihood")

        n = len(self.y_train)

        try:
            log_det = 2.0 * np.sum(np.log(np.diag(self.L)))
        except AttributeError:
            sign, log_det = np.linalg.slogdet(self.K)
            if sign <= 0:
                return -np.inf

        return float(
            -0.5 * self.y_train @ self.alpha
            -0.5 * log_det
            -0.5 * n * np.log(2.0 * np.pi)
        )

    def optimize_hyperparameters(self):
        # Basic deterministic grid search over common scale parameters.
        if not hasattr(self, "X_train"):
            raise ValueError("Call fit before optimizing hyperparameters")

        if self.kernel == "linear":
            return self

        if self.kernel == "rbf":
            candidates = [0.1, 0.3, 1.0, 3.0, 10.0]

            best_params = dict(self.kernel_params)
            best_score = -np.inf

            original = dict(self.kernel_params)

            for ls in candidates:
                self.kernel_params = dict(original)
                self.kernel_params["length_scale"] = ls

                if "sigma" in original:
                    self.kernel_params["sigma"] = original["sigma"]

                self.fit(self.X_train, self.y_train)
                score = self.log_marginal_likelihood()

                if score > best_score:
                    best_score = score
                    best_params = dict(self.kernel_params)

            self.kernel_params = best_params
            self.fit(self.X_train, self.y_train)

        return self