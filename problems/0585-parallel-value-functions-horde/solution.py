import numpy as np


def horde_parallel_gvfs(
    experience: list,
    demons: list,
    behavior_policy: np.ndarray,
    n_states: int,
    n_actions: int,
    alpha: float
) -> list:
    """
    Learn multiple General Value Functions in parallel from a single
    experience stream using off-policy TD(lambda) with importance sampling.
    """

    behavior_policy = np.asarray(behavior_policy, dtype=float)

    # Each demon has its own value function and eligibility trace.
    values = [
        np.zeros(n_states, dtype=float)
        for _ in demons
    ]

    traces = [
        np.zeros(n_states, dtype=float)
        for _ in demons
    ]

    for state, action, next_state in experience:
        state = int(state)
        action = int(action)
        next_state = int(next_state)

        # Behavior probability for the action actually taken.
        mu = behavior_policy[state, action]

        for i, demon in enumerate(demons):
            gamma = float(demon["gamma"])
            lambd = float(demon["lambd"])

            target_policy = np.asarray(
                demon["target_policy"],
                dtype=float
            )
            cumulant = np.asarray(
                demon["cumulant"],
                dtype=float
            )

            # Importance sampling ratio.
            if mu == 0.0:
                rho = 0.0
            else:
                rho = target_policy[state, action] / mu

            # Cumulant for this transition.
            c = float(cumulant[state, action, next_state])

            # Current and next value estimates.
            v = values[i][state]
            v_next = values[i][next_state]

            # TD error.
            delta = c + gamma * v_next - v

            # Importance-weighted accumulating trace:
            # 1. decay old trace
            # 2. add current-state indicator
            # 3. scale entire trace by rho
            traces[i] *= gamma * lambd
            traces[i][state] += 1.0
            traces[i] *= rho

            # Value update.
            values[i] += alpha * delta * traces[i]

    return [
        [round(float(v), 4) for v in value]
        for value in values
    ]