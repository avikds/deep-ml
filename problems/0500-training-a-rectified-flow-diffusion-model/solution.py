import numpy as np

def rectified_flow_step(
    X_1: np.ndarray,
    X_0: np.ndarray,
    t: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    lr: float
) -> dict:
    """
    Perform one training step for a rectified flow model.
    """

    X_1 = np.asarray(X_1, dtype=float)
    X_0 = np.asarray(X_0, dtype=float)
    t = np.asarray(t, dtype=float)
    W = np.asarray(W, dtype=float)
    b = np.asarray(b, dtype=float)

    # 1. Straight-line interpolation:
    #    t=0 -> noise X_0
    #    t=1 -> data  X_1
    x_t = (1.0 - t) * X_0 + t * X_1

    # 2. Ground-truth velocity
    v_target = X_1 - X_0

    # 3. Linear velocity network
    #    Input is [x_t | t]
    network_input = np.concatenate([x_t, t], axis=1)
    v_pred = network_input @ W + b

    # 4. Mean squared error over all N*D elements
    error = v_pred - v_target
    loss = float(np.mean(error ** 2))

    # 5. Analytical gradients
    N, D = v_target.shape
    scale = 2.0 / (N * D)

    grad_W = scale * (network_input.T @ error)
    grad_b = scale * np.sum(error, axis=0, keepdims=True)

    # 6. Gradient descent update
    W_new = W - lr * grad_W
    b_new = b - lr * grad_b

    return {
        "x_t": x_t,
        "v_target": v_target,
        "v_pred": v_pred,
        "loss": loss,
        "W_new": W_new,
        "b_new": b_new,
    }