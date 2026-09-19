import numpy as np

def semi_markov_q_learning(
    episodes: list,
    n_states: int,
    n_actions: int,
    gamma: float,
    alpha: float
) -> list:
    """
    Implement Semi-Markov Q-Learning with variable action durations.
    """
    Q = np.zeros((n_states, n_actions), dtype=float)

    for episode in episodes:
        for state, action, reward, next_state, duration, done in episode:
            if done:
                target = reward
            else:
                target = reward + (gamma ** duration) * np.max(Q[next_state])

            Q[state, action] += alpha * (
                target - Q[state, action]
            )

    return np.round(Q, 4).tolist()