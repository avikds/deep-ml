import numpy as np

def true_online_td_lambda(episodes, n_features, alpha, gamma, lam):
    """
    True Online TD(lambda) for policy evaluation with linear function approximation.
    
    Args:
        episodes: List of episodes, each a list of
                  (state_features, reward, next_state_features, done)
        n_features: Dimensionality of feature vectors
        alpha: Step size
        gamma: Discount factor
        lam: Trace decay parameter
    
    Returns:
        Final weight vector as a numpy array of shape (n_features,)
    """
    # Initialize weights.
    w = np.zeros(n_features, dtype=float)

    for episode in episodes:
        # Reset true-online trace and previous value at each episode.
        z = np.zeros(n_features, dtype=float)
        v_old = 0.0

        for state_features, reward, next_state_features, done in episode:
            x = np.asarray(state_features, dtype=float)
            x_next = np.asarray(next_state_features, dtype=float)

            # Current and next value estimates.
            v = np.dot(w, x)

            if done:
                v_next = 0.0
            else:
                v_next = np.dot(w, x_next)

            # TD error.
            delta = reward + gamma * v_next - v

            # Dutch eligibility trace.
            z = (
                gamma * lam * z
                + (1.0 - alpha * gamma * lam * np.dot(z, x)) * x
            )

            # True Online TD(lambda) weight correction.
            w += (
                alpha * (delta + v - v_old) * z
                - alpha * (v - v_old) * x
            )

            # Current next-state estimate becomes old value.
            v_old = v_next

    return w