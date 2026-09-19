import numpy as np


def linear_sarsa(
    episodes: list,
    features: dict,
    n_features: int,
    alpha: float,
    gamma: float
) -> list:
    """
    Episodic semi-gradient Sarsa with linear function approximation.
    """

    # Initialize weight vector.
    w = np.zeros(n_features, dtype=float)

    for episode in episodes:
        for t, (state, action, reward) in enumerate(episode):
            # Current state-action features.
            x = np.asarray(features[(state, action)], dtype=float)

            # Current Q estimate.
            q_current = np.dot(w, x)

            # Last transition is terminal.
            if t == len(episode) - 1:
                q_next = 0.0
            else:
                next_state, next_action, _ = episode[t + 1]

                x_next = np.asarray(
                    features[(next_state, next_action)],
                    dtype=float
                )

                q_next = np.dot(w, x_next)

            # Sarsa TD error.
            delta = reward + gamma * q_next - q_current

            # Semi-gradient update.
            w += alpha * delta * x

    return [round(float(v), 4) for v in w]