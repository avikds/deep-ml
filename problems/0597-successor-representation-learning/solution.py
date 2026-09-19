import numpy as np

def learn_successor_representation(
    experience: list,
    n_states: int,
    gamma: float,
    alpha_sr: float,
    alpha_w: float
) -> tuple:
    """
    Learn the Successor Representation from a stream of experience.
    """

    M = np.zeros((n_states, n_states), dtype=float)
    w = np.zeros(n_states, dtype=float)

    for state, reward, next_state, done in experience:
        # Reward-weight update.
        w[state] += alpha_w * (reward - w[state])

        # SR TD target:
        # e_s + gamma * M[s'] for non-terminal transitions.
        target = np.zeros(n_states, dtype=float)
        target[state] = 1.0

        if not done:
            target += gamma * M[next_state]

        # Update only the current state's SR row.
        M[state] += alpha_sr * (target - M[state])

    # Reconstruct value function from SR and learned reward weights.
    V = M @ w

    return (
        np.round(M, 4).tolist(),
        np.round(w, 4).tolist(),
        np.round(V, 4).tolist()
    )