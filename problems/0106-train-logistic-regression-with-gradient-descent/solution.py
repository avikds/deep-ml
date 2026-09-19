import numpy as np

def train_logreg(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float,
    iterations: int
) -> tuple[list[float], list[float]]:

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    m, n = X.shape

    # Add bias column
    X_bias = np.column_stack((np.ones(m), X))

    # Initialize weights to zero
    theta = np.zeros(n + 1, dtype=float)

    losses = []

    for _ in range(iterations):
        # Forward pass
        z = X_bias @ theta
        z = np.clip(z, -500, 500)
        predictions = 1.0 / (1.0 + np.exp(-z))

        # Sum Binary Cross Entropy loss
        eps = 1e-15
        loss = -np.sum(
            y * np.log(np.clip(predictions, eps, 1.0 - eps))
            + (1.0 - y) * np.log(np.clip(1.0 - predictions, eps, 1.0))
        )

        # Record loss BEFORE the update
        losses.append(round(float(loss), 4))

        # Sum-based gradient
        gradient = X_bias.T @ (predictions - y)

        # Gradient descent
        theta -= learning_rate * gradient

    return theta.tolist(), losses