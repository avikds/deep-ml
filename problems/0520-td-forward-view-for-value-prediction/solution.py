import numpy as np


def td_lambda_forward_view(
    episodes: list,
    n_states: int,
    gamma: float,
    alpha: float,
    lam: float
) -> list:
    """
    Implement forward-view TD(lambda) prediction.

    Args:
        episodes: List of episodes, each a list of
                  (state, reward, next_state, done) tuples
        n_states: Number of states
        gamma: Discount factor
        alpha: Learning rate
        lam: Lambda parameter for trace decay

    Returns:
        List of estimated state values rounded to 4 decimal places
    """

    # 1. Initialize value function.
    V = np.zeros(n_states, dtype=float)

    # 2. Process episodes in order.
    for episode in episodes:
        T = len(episode)

        if T == 0:
            continue

        # Snapshot of V at the START of this episode.
        V_snapshot = V.copy()

        lambda_returns = np.zeros(T, dtype=float)

        # 3. Compute all lambda-returns using the snapshot.
        for t in range(T):
            # Build all n-step returns from timestep t.
            n_step_returns = []
            cumulative_reward = 0.0
            discount = 1.0

            for n in range(1, T - t + 1):
                state, reward, next_state, done = episode[t + n - 1]

                cumulative_reward += discount * reward

                if done:
                    # Terminal transition: no bootstrap.
                    G_n = cumulative_reward
                    n_step_returns.append(G_n)
                    break

                # If this is the longest available return, bootstrap
                # from the state reached after the final transition.
                if n == T - t:
                    G_n = cumulative_reward + discount * gamma * V_snapshot[
                        next_state
                    ]
                else:
                    G_n = cumulative_reward + discount * gamma * V_snapshot[
                        next_state
                    ]

                n_step_returns.append(G_n)
                discount *= gamma

            # Lambda weighting.
            H = len(n_step_returns)

            if H == 1:
                G_lambda = n_step_returns[0]
            else:
                G_lambda = 0.0

                # n = 1, ..., H-1:
                # weight = (1-lambda) * lambda^(n-1)
                for n in range(1, H):
                    weight = (1.0 - lam) * (lam ** (n - 1))
                    G_lambda += weight * n_step_returns[n - 1]

                # Longest return gets remaining weight.
                G_lambda += (lam ** (H - 1)) * n_step_returns[H - 1]

            lambda_returns[t] = G_lambda

        # 4. Sequentially update V, but use the episode-start
        # snapshot in the error term.
        for t in range(T):
            state = episode[t][0]

            V[state] += alpha * (
                lambda_returns[t] - V_snapshot[state]
            )

    return [round(float(v), 4) for v in V]