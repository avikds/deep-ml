import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute the average Huber Loss between true and predicted values.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    error = np.abs(y_true - y_pred)

    loss = np.where(
        error <= delta,
        0.5 * error**2,
        delta * (error - 0.5 * delta)
    )

    return float(np.mean(loss))