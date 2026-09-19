import numpy as np

def td_lambda_convergence(
    episodes: list,
    n_states: int,
    gamma: float,
    lam: float,
    alpha: float,
    true_values: np.ndarray,
    tolerance: float
) -> tuple:
    """
    Run TD(lambda) with accumulating eligibility traces and monitor convergence.
    """
    v = np.zeros(n_states, dtype=float)
    true_values = np.asarray(true_values, dtype=float)

    rmse_history = []
    convergence_episode = -1

    for episode_idx, episode in enumerate(episodes):
        # Fresh eligibility traces for every episode.
        e = np.zeros(n_states, dtype=float)

        for state, reward, next_state, done in episode:
            # TD error using the current value estimates.
            if done:
                next_value = 0.0
            else:
                next_value = v[next_state]

            delta = reward + gamma * next_value - v[state]

            # Accumulating eligibility trace.
            e *= gamma * lam
            e[state] += 1.0

            # Backward-view TD(lambda) update.
            v += alpha * delta * e

        # RMSE after the complete episode.
        rmse = np.sqrt(np.mean((v - true_values) ** 2))
        rmse_history.append(rmse)

        if convergence_episode == -1 and rmse < tolerance:
            convergence_episode = episode_idx

    return (
        np.round(v, 4).tolist(),
        np.round(rmse_history, 4).tolist(),
        int(convergence_episode)
    )