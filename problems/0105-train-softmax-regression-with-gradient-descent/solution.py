import numpy as np

def train_softmaxreg(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float,
    iterations: int
) -> tuple[list[float], list[float]]:

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    n_samples = X.shape[0]
    n_features = X.shape[1]
    n_classes = int(np.max(y)) + 1

    # Add bias column
    X_bias = np.column_stack((np.ones(n_samples), X))

    # Initialize coefficients to zero
    coefficients = np.zeros((n_classes, n_features + 1), dtype=float)

    # One-hot encoded targets
    Y = np.zeros((n_samples, n_classes), dtype=float)
    Y[np.arange(n_samples), y] = 1.0

    losses = []

    for _ in range(iterations):
        # Scores
        scores = X_bias @ coefficients.T

        # Numerically stable softmax
        scores -= np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # Sum cross-entropy loss
        loss = -np.sum(
            Y * np.log(np.clip(probs, 1e-15, 1.0))
        )

        # Sum-based gradient
        # dL/dW = (P - Y)^T X
        gradient = (probs - Y).T @ X_bias

        # Gradient descent update
        coefficients -= learning_rate * gradient

        # Record loss after the weight update, as requested
        losses.append(float(loss))

    return coefficients.tolist(), losses