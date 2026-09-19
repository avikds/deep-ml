import numpy as np

def semi_gradient_td_lambda(
    episodes: list,
    n_features: int,
    alpha: float,
    gamma: float,
    lam: float,
    initial_w: list = None
) -> list:
    """
    Implement semi-gradient TD(lambda) with accumulating eligibility traces
    and linear function approximation.
    """

    # Initialize weights.
    if initial_w is None:
        w = np.zeros(n_features, dtype=float)
    else:
        w = np.asarray(initial_w, dtype=float).copy()

    for episode in episodes:
        # Reset eligibility traces for every episode.
        z = np.zeros(n_features, dtype=float)

        for state_features, reward, next_state_features in episode:
            x = np.asarray(state_features, dtype=float)
            reward = float(reward)

            # Current value estimate.
            value = np.dot(w, x)

            # Bootstrap from next state unless terminal.
            if next_state_features is None:
                next_value = 0.0
            else:
                x_next = np.asarray(next_state_features, dtype=float)
                next_value = np.dot(w, x_next)

            # TD error.
            delta = reward + gamma * next_value - value

            # Accumulating eligibility trace.
            z = gamma * lam * z + x

            # Semi-gradient TD(lambda) update.
            w += alpha * delta * z

    return [round(float(v), 4) for v in w]