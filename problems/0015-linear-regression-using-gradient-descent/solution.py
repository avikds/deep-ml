import numpy as np

def linear_regression_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    iterations: int
) -> np.ndarray:
    """
    Perform linear regression using batch gradient descent.
    """
    m, n = X.shape
    y = y.reshape(-1, 1)
    theta = np.zeros((n, 1))

    for _ in range(iterations):
        predictions = X @ theta
        errors = predictions - y

        # Gradient of (1 / 2m) * sum(errors^2)
        gradient = (X.T @ errors) / m

        theta -= alpha * gradient

    return theta.flatten()