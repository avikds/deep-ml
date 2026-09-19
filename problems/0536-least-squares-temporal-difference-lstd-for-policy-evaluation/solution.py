import numpy as np


def lstd(features, next_features, rewards, gamma, epsilon=0.01):
    """
    Least-Squares Temporal Difference (LSTD) for policy evaluation.
    
    Args:
        features: array-like of shape (T, d) - feature vectors for each state
        next_features: array-like of shape (T, d) - feature vectors for next states
        rewards: array-like of length T - observed rewards
        gamma: float - discount factor
        epsilon: float - regularization constant
    
    Returns:
        List of floats - weight vector w of length d
    """
    features = np.asarray(features, dtype=float)
    next_features = np.asarray(next_features, dtype=float)
    rewards = np.asarray(rewards, dtype=float)

    T, d = features.shape

    # A = epsilon * I + sum(phi_t (phi_t - gamma * phi_{t+1})^T)
    A = epsilon * np.eye(d)

    # b = sum(phi_t * reward_t)
    b = np.zeros(d, dtype=float)

    for t in range(T):
        phi = features[t]
        phi_next = next_features[t]

        A += np.outer(
            phi,
            phi - gamma * phi_next
        )

        b += phi * rewards[t]

    # Solve A w = b
    w = np.linalg.solve(A, b)

    return w.tolist()