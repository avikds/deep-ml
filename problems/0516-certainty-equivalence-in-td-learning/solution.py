import numpy as np


def certainty_equivalence_td(episodes, n_states, gamma):
    """
    Compute the certainty-equivalence value function from observed transition data.

    Args:
        episodes: list of episodes, each episode is a list of
                  (state, reward, next_state) tuples
        n_states: int, number of states in the MDP
        gamma: float, discount factor

    Returns:
        np.ndarray: value function of shape (n_states,)
    """

    # Transition counts and reward sums for each source state.
    transition_counts = np.zeros((n_states, n_states), dtype=float)
    reward_sums = np.zeros(n_states, dtype=float)
    state_counts = np.zeros(n_states, dtype=float)

    # ---------------------------------------------------------
    # 1. Estimate the MDP from the observed transitions.
    # ---------------------------------------------------------
    for episode in episodes:
        for state, reward, next_state in episode:
            state = int(state)
            next_state = int(next_state)

            transition_counts[state, next_state] += 1.0
            reward_sums[state] += float(reward)
            state_counts[state] += 1.0

    # Empirical transition probabilities and expected rewards.
    P_hat = np.zeros((n_states, n_states), dtype=float)
    R_hat = np.zeros(n_states, dtype=float)

    visited = state_counts > 0

    P_hat[visited] = (
        transition_counts[visited]
        / state_counts[visited, None]
    )

    R_hat[visited] = (
        reward_sums[visited]
        / state_counts[visited]
    )

    # ---------------------------------------------------------
    # 2. Solve the Bellman linear system:
    #
    #     V = R_hat + gamma * P_hat @ V
    #
    #     (I - gamma P_hat)V = R_hat
    # ---------------------------------------------------------
    A = np.eye(n_states) - gamma * P_hat

    # States never observed as source states must have V = 0.
    # Their corresponding equations are simply V[s] = 0.
    for state in range(n_states):
        if not visited[state]:
            A[state, :] = 0.0
            A[state, state] = 1.0
            R_hat[state] = 0.0

    values = np.linalg.solve(A, R_hat)

    return values