import numpy as np

def train(X, y, W, b):
    """
    Train linear regression weights on standardized data.

    Args:
        X: numpy array of shape (n_samples, n_features)
        y: numpy array of shape (n_samples,)
        W: numpy array of shape (n_features,)
        b: float

    Returns:
        W: numpy array of shape (n_features,)
        b: float
    """
    X = np.asarray(X)
    y = np.asarray(y)

    # Add a bias column: [X, 1]
    X_aug = np.column_stack((X, np.ones(X.shape[0])))

    # Solve the least-squares problem:
    # min ||X_aug @ theta - y||^2
    theta, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)

    W = np.asarray(theta[:-1], dtype=float)
    b = float(theta[-1])

    return W, b