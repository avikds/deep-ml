import numpy as np


def td_gammon_learning(game_states, outcome, W1, b1, w2, b2, alpha, lambd):
    """
    Apply TD(lambda) learning to update a neural network value function
    over a single self-play game sequence.
    """

    # Work on copies so the input parameters are not modified unexpectedly.
    W1 = np.asarray(W1, dtype=float).copy()
    b1 = np.asarray(b1, dtype=float).copy()
    w2 = np.asarray(w2, dtype=float).copy()
    b2 = float(b2)

    # Eligibility traces for each parameter tensor/vector.
    e_W1 = np.zeros_like(W1)
    e_b1 = np.zeros_like(b1)
    e_w2 = np.zeros_like(w2)
    e_b2 = 0.0

    def sigmoid(x):
        x = np.clip(x, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-x))

    def forward(x):
        z1 = W1 @ x + b1
        h = sigmoid(z1)
        z2 = np.dot(w2, h) + b2
        V = sigmoid(z2)
        return V, z1, h

    for t in range(len(game_states)):
        x = np.asarray(game_states[t], dtype=float)

        # Current-state forward pass.
        V_t, z1, h = forward(x)

        # Compute the next-state value BEFORE updating parameters.
        if t + 1 < len(game_states):
            x_next = np.asarray(game_states[t + 1], dtype=float)
            V_next, _, _ = forward(x_next)
            delta = V_next - V_t
        else:
            delta = float(outcome) - V_t

        # ---------------------------------------------------------
        # Backpropagation: gradient of V_t wrt all parameters.
        # ---------------------------------------------------------

        # dV/dz2
        output_derivative = V_t * (1.0 - V_t)

        # dV/dw2
        grad_w2 = output_derivative * h

        # dV/db2
        grad_b2 = output_derivative

        # Backprop into hidden layer.
        dV_dh = output_derivative * w2

        # dh/dz1
        hidden_derivative = h * (1.0 - h)

        # dV/dz1
        dV_dz1 = dV_dh * hidden_derivative

        # dV/dW1
        grad_W1 = np.outer(dV_dz1, x)

        # dV/db1
        grad_b1 = dV_dz1

        # ---------------------------------------------------------
        # Eligibility traces:
        # e_t = lambda * e_{t-1} + grad V(s_t)
        # ---------------------------------------------------------
        e_W1 = lambd * e_W1 + grad_W1
        e_b1 = lambd * e_b1 + grad_b1
        e_w2 = lambd * e_w2 + grad_w2
        e_b2 = lambd * e_b2 + grad_b2

        # ---------------------------------------------------------
        # TD(lambda) parameter update.
        # ---------------------------------------------------------
        W1 += alpha * delta * e_W1
        b1 += alpha * delta * e_b1
        w2 += alpha * delta * e_w2
        b2 += alpha * delta * e_b2

    return W1, b1, w2, float(b2)