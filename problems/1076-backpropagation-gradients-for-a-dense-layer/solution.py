import numpy as np

def dense_backward(
    a_prev: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    y: np.ndarray
) -> dict:
    """
    Compute gradients of the squared-error cost w.r.t. W, b, and a_prev
    for a dense layer z = W @ a_prev + b followed by sigmoid activation.
    """
    z = W @ a_prev + b
    a = 1.0 / (1.0 + np.exp(-z))

    # dC/da = 2(a - y)
    # da/dz = a(1-a)
    dz = 2.0 * (a - y) * a * (1.0 - a)

    # dC/dW = dz[:, None] * a_prev[None, :]
    dW = np.outer(dz, a_prev)

    # dC/db = dz
    db = dz

    # dC/da_prev = W^T @ dz
    da_prev = W.T @ dz

    return {
        "dW": dW.tolist(),
        "db": db.tolist(),
        "da_prev": da_prev.tolist()
    }