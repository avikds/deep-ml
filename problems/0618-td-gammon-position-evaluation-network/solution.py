import numpy as np

def td_position_eval(
    states: list,
    outcome: float,
    W1: list,
    b1: list,
    W2: list,
    b2: list,
    alpha: float,
    gamma: float,
    lam: float
) -> dict:
    """
    Train a position evaluation network using online TD(lambda).
    """
    W1 = np.asarray(W1, dtype=float).copy()
    b1 = np.asarray(b1, dtype=float).copy()
    W2 = np.asarray(W2, dtype=float).copy()
    b2 = np.asarray(b2, dtype=float).copy()

    # Eligibility traces for each parameter.
    e_W1 = np.zeros_like(W1)
    e_b1 = np.zeros_like(b1)
    e_W2 = np.zeros_like(W2)
    e_b2 = np.zeros_like(b2)

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))

    def forward_and_grad(x):
        # Forward pass.
        z1 = W1 @ x + b1
        h = sigmoid(z1)

        z2 = float((W2 @ h)[0] + b2[0])
        v = float(sigmoid(z2))

        # dv/dz2.
        dz2 = v * (1.0 - v)

        # Gradients of v with respect to output parameters.
        grad_W2 = dz2 * h[None, :]
        grad_b2 = np.array([dz2], dtype=float)

        # Backpropagate through hidden layer.
        dz1 = dz2 * W2[0] * h * (1.0 - h)

        grad_W1 = dz1[:, None] * x[None, :]
        grad_b1 = dz1.copy()

        return v, grad_W1, grad_b1, grad_W2, grad_b2

    T = len(states)

    for t in range(T):
        x = np.asarray(states[t], dtype=float)

        # Current value and its gradient, using current parameters.
        v_t, grad_W1, grad_b1, grad_W2, grad_b2 = forward_and_grad(x)

        # Eligibility trace update.
        decay = gamma * lam
        e_W1 = decay * e_W1 + grad_W1
        e_b1 = decay * e_b1 + grad_b1
        e_W2 = decay * e_W2 + grad_W2
        e_b2 = decay * e_b2 + grad_b2

        # TD target.
        if t == T - 1:
            target = outcome
        else:
            next_x = np.asarray(states[t + 1], dtype=float)

            # Next value is computed with the current weights,
            # before applying this timestep's update.
            z1_next = W1 @ next_x + b1
            h_next = sigmoid(z1_next)
            z2_next = float((W2 @ h_next)[0] + b2[0])
            v_next = float(sigmoid(z2_next))

            target = gamma * v_next

        delta = target - v_t

        # Online TD(lambda) parameter update.
        W1 += alpha * delta * e_W1
        b1 += alpha * delta * e_b1
        W2 += alpha * delta * e_W2
        b2 += alpha * delta * e_b2

    return {
        "W1": np.round(W1, 4).tolist(),
        "b1": np.round(b1, 4).tolist(),
        "W2": np.round(W2, 4).tolist(),
        "b2": np.round(b2, 4).tolist()
    }