import numpy as np


def r_learning(
    transitions: list,
    n_states: int,
    n_actions: int,
    alpha: float,
    beta: float,
    initial_rho: float = 0.0
) -> tuple:
    """
    R-Learning algorithm for average reward MDPs.
    """

    # Initialize differential action values and average reward.
    Q = np.zeros((n_states, n_actions), dtype=float)
    rho = float(initial_rho)

    for state, action, reward, next_state in transitions:
        state = int(state)
        action = int(action)
        next_state = int(next_state)
        reward = float(reward)

        # Differential TD error.
        delta = (
            reward
            - rho
            + np.max(Q[next_state])
            - Q[state, action]
        )

        # Q-value update.
        Q[state, action] += alpha * delta

        # Greedy action according to the UPDATED Q-values.
        # np.argmax breaks ties in favor of the smallest index.
        greedy_action = int(np.argmax(Q[state]))

        # Update average reward only for greedy actions.
        if action == greedy_action:
            rho += beta * delta

    return (
        [[round(float(x), 4) for x in row] for row in Q],
        round(float(rho), 4)
    )