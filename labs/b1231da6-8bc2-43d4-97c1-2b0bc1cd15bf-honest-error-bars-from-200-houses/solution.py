import numpy as np
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    Matern,
    WhiteKernel,
    DotProduct
)


def _features(X):
    X = np.asarray(X, dtype=np.float64).copy()

    # Compress the strongly right-skewed variables.
    for j in (2, 3, 4, 5):
        X[:, j] = np.log1p(np.maximum(X[:, j], 0.0))

    # Handle any unexpected missing or non-finite values.
    for j in range(X.shape[1]):
        col = X[:, j]
        valid = np.isfinite(col)
        med = np.median(col[valid]) if np.any(valid) else 0.0
        col[~valid] = med

    return X


def _optimizer(obj_func, initial_theta, bounds):
    # Limit optimization work: each task has only 200 observations.
    result = minimize(
        obj_func,
        initial_theta,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={
            "maxiter": 35,
            "ftol": 1e-7
        }
    )
    return result.x, result.fun


def fit_predict(X_train, y_train, X_test):
    """
    Fit a Gaussian process and return predictive means and variances.
    """
    Xtr = _features(X_train)
    Xte = _features(X_test)
    y = np.asarray(y_train, dtype=np.float64)

    # Standardize inputs.
    scaler = StandardScaler()
    A = scaler.fit_transform(Xtr)
    B = scaler.transform(Xte)

    # Standardize the regression target.
    y_mean = float(np.mean(y))
    y_std = float(np.std(y))

    if y_std < 1e-8:
        y_std = 1.0

    ys = (y - y_mean) / y_std

    # Nonlinear ARD Matérn component.
    nonlinear = (
        ConstantKernel(1.0, (0.05, 20.0))
        * Matern(
            length_scale=np.ones(A.shape[1]),
            length_scale_bounds=(0.12, 12.0),
            nu=2.5
        )
    )

    # Linear component captures broad income and geographic trends.
    linear = (
        ConstantKernel(0.3, (0.01, 5.0))
        * DotProduct(
            sigma_0=1.0,
            sigma_0_bounds=(0.1, 10.0)
        )
    )

    # Explicit observation noise.
    noise = WhiteKernel(
        noise_level=0.15,
        noise_level_bounds=(0.003, 1.5)
    )

    kernel = nonlinear + linear + noise

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        optimizer=_optimizer,
        n_restarts_optimizer=0,
        normalize_y=False,
        copy_X_train=True
    )

    gp.fit(A, ys)

    # Predictive mean and standard deviation.
    mean_scaled, std_scaled = gp.predict(
        B,
        return_std=True
    )

    mean = y_mean + y_std * mean_scaled

    # Convert variance back into the original target units.
    var = (y_std * std_scaled) ** 2

    # Numerical floor ensures strictly positive predictive variance.
    var = np.maximum(var, 1e-4)

    return (
        np.asarray(mean, dtype=np.float64),
        np.asarray(var, dtype=np.float64)
    )