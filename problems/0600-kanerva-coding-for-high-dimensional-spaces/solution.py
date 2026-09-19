import numpy as np

def kanerva_coding_td(
    prototypes: np.ndarray,
    threshold: float,
    episodes: list,
    gamma: float,
    alpha: float,
    query_states: np.ndarray
) -> tuple:
    """
    Kanerva coding with semi-gradient TD(0) for value function approximation.
    """
    prototypes = np.asarray(prototypes, dtype=float)
    query_states = np.asarray(query_states, dtype=float)

    k = prototypes.shape[0]
    weights = np.zeros(k, dtype=float)

    def features(state):
        state = np.asarray(state, dtype=float)
        distances = np.linalg.norm(prototypes - state, axis=1)
        return (distances <= threshold).astype(float)

    def value(state):
        phi = features(state)
        return float(np.dot(weights, phi))

    # Online semi-gradient TD(0).
    for episode in episodes:
        for state, reward, next_state, done in episode:
            phi = features(state)
            current_value = float(np.dot(weights, phi))

            if done:
                target = float(reward)
            else:
                target = float(reward) + gamma * value(next_state)

            td_error = target - current_value

            weights += alpha * td_error * phi

    # Evaluate query states using the final weights.
    values = []
    for state in query_states:
        values.append(value(state))

    return (
        np.round(weights, 4).tolist(),
        np.round(values, 4).tolist()
    )