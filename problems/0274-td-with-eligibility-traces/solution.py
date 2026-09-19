import numpy as np

def td_lambda_prediction(
    episodes: list[list[tuple[int, float]]],
    n_states: int,
    gamma: float,
    lambd: float,
    alpha: float
) -> np.ndarray:
    """
    Estimate state values using TD(lambda) with accumulating eligibility traces.
    """
    V = np.zeros(n_states, dtype=float)

    for episode in episodes:
        e = np.zeros(n_states, dtype=float)

        for t, (state, reward) in enumerate(episode):
            # Decay existing traces, then accumulate the current state's trace.
            e *= gamma * lambd
            e[state] += 1.0

            # Bootstrap from the next state's value when it exists.
            if t + 1 < len(episode):
                next_state = episode[t + 1][0]
                delta = reward + gamma * V[next_state] - V[state]
            else:
                # Terminal transition has no bootstrap value.
                delta = reward - V[state]

            # Update every state proportionally to its eligibility.
            V += alpha * delta * e

    return V