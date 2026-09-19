import numpy as np


def residual_gradient_td(episodes, features, gamma, alpha, n_passes):
    """
    Run the Residual Gradient algorithm for policy evaluation
    with linear function approximation.
    
    Args:
        episodes: list of episodes, each is a list of
                  (state, reward, next_state) tuples.
                  next_state = -1 indicates terminal.
        features: np.ndarray of shape (n_states, d),
                  feature vectors per state.
        gamma: float, discount factor.
        alpha: float, step size.
        n_passes: int, number of passes through the data.
    
    Returns:
        np.ndarray of shape (d,): learned weight vector.
    """
    features = np.asarray(features, dtype=float)

    n_states, d = features.shape
    w = np.zeros(d, dtype=float)

    for _ in range(n_passes):
        for episode in episodes:
            for state, reward, next_state in episode:
                state = int(state)
                reward = float(reward)
                next_state = int(next_state)

                # Current-state feature vector.
                phi = features[state]

                # Terminal transitions have zero next-state value
                # and zero next-state feature vector.
                if next_state == -1:
                    phi_next = np.zeros(d, dtype=float)
                    v_next = 0.0
                else:
                    phi_next = features[next_state]
                    v_next = np.dot(w, phi_next)

                # Current value estimate.
                v = np.dot(w, phi)

                # Full TD error.
                delta = reward + gamma * v_next - v

                # Residual-gradient update.
                gradient_direction = phi - gamma * phi_next
                w += alpha * delta * gradient_direction

    return w