import numpy as np

def bellman_error_minimization(
    features: np.ndarray,
    transition_probs: np.ndarray,
    rewards: np.ndarray,
    gamma: float,
    state_distribution: np.ndarray,
    learning_rate: float,
    n_iterations: int,
    initial_weights: np.ndarray = None
) -> tuple:
    """
    Minimize the Mean Squared Bellman Error using gradient descent
    with linear function approximation.
    """
    features = np.asarray(features, dtype=float)
    transition_probs = np.asarray(transition_probs, dtype=float)
    rewards = np.asarray(rewards, dtype=float)
    state_distribution = np.asarray(state_distribution, dtype=float)

    n_features = features.shape[1]

    if initial_weights is None:
        weights = np.zeros(n_features, dtype=float)
    else:
        weights = np.asarray(initial_weights, dtype=float).copy()

    # Bellman-error gradient:
    # delta = V - (r + gamma P V)
    # grad(MSBE) = 2 Phi^T D delta, accounting for the dependence
    # of the bootstrap value on w:
    # delta = (I - gamma P) Phi w - r
    A = (np.eye(features.shape[0]) - gamma * transition_probs) @ features

    for _ in range(n_iterations):
        values = features @ weights
        bellman_target = rewards + gamma * (transition_probs @ values)
        delta = values - bellman_target

        # Exact gradient of the MSBE:
        # 2 * A^T D delta
        gradient = 2.0 * (A.T @ (state_distribution * delta))

        weights -= learning_rate * gradient

    # Final values and MSBE.
    values = features @ weights
    bellman_target = rewards + gamma * (transition_probs @ values)
    delta = values - bellman_target
    msbe = float(np.sum(state_distribution * delta ** 2))

    return (
        np.round(weights, 4).tolist(),
        round(msbe, 4),
        np.round(values, 4).tolist()
    )