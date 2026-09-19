import numpy as np

def true_online_sarsa_lambda(episodes, n_features, alpha, gamma, lam):
    """
    Implement True Online SARSA(lambda) with linear function approximation.

    Args:
        episodes: list of episodes. Each episode is a list of tuples
                  (x, reward, x_next, done) where x and x_next are
                  numpy arrays of shape (n_features,).
        n_features: int, dimension of feature vectors
        alpha: float, step size
        gamma: float, discount factor
        lam: float, trace decay parameter

    Returns:
        list of floats: learned weight vector rounded to 5 decimal places
    """

    # Initialize weight vector.
    w = np.zeros(n_features, dtype=float)

    for episode in episodes:
        # Reset eligibility trace and Q_old for each episode.
        z = np.zeros(n_features, dtype=float)
        q_old = 0.0

        for x, reward, x_next, done in episode:
            x = np.asarray(x, dtype=float)
            x_next = np.asarray(x_next, dtype=float)

            # Current estimate Q(S,A).
            q = np.dot(w, x)

            # Next-state/action estimate.
            if done:
                q_next = 0.0
            else:
                q_next = np.dot(w, x_next)

            # TD error.
            delta = reward + gamma * q_next - q

            # Dutch eligibility trace.
            z = (
                gamma * lam * z
                + (1.0 - alpha * gamma * lam * np.dot(z, x)) * x
            )

            # True Online SARSA(lambda) weight update.
            w += (
                alpha * (delta + q - q_old) * z
                - alpha * (q - q_old) * x
            )

            # Store next-state value for the following step.
            q_old = q_next

        # q_old is reset automatically on the next episode.

    return [round(float(v), 5) for v in w]