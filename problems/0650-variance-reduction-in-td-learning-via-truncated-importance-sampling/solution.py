import numpy as np

def variance_reduced_td(
    episodes: list,
    behavior_policy: list,
    target_policy: list,
    num_states: int,
    num_actions: int,
    alpha: float,
    gamma: float,
    c_bar: float
) -> dict:
    """
    Off-policy TD(0) prediction with standard and truncated importance sampling.
    """
    behavior_policy = np.asarray(behavior_policy, dtype=float)
    target_policy = np.asarray(target_policy, dtype=float)

    V_standard = np.zeros(num_states, dtype=float)
    V_truncated = np.zeros(num_states, dtype=float)

    # Per-state collections of weighted TD errors.
    weighted_standard = [[] for _ in range(num_states)]
    weighted_truncated = [[] for _ in range(num_states)]

    for episode in episodes:
        T = len(episode)

        for t, (state, action, reward) in enumerate(episode):
            # Importance sampling ratio.
            b = behavior_policy[state, action]
            pi = target_policy[state, action]

            if b == 0.0:
                rho = 0.0
            else:
                rho = pi / b

            truncated_rho = min(rho, c_bar)

            # Terminal transition has zero next-state value.
            if t == T - 1:
                next_value_standard = 0.0
                next_value_truncated = 0.0
            else:
                next_state = episode[t + 1][0]
                next_value_standard = V_standard[next_state]
                next_value_truncated = V_truncated[next_state]

            # Standard IS TD error and weighted update signal.
            delta_standard = (
                reward
                + gamma * next_value_standard
                - V_standard[state]
            )
            weighted_standard_error = rho * delta_standard

            V_standard[state] += alpha * weighted_standard_error
            weighted_standard[state].append(weighted_standard_error)

            # Truncated IS TD error and weighted update signal.
            delta_truncated = (
                reward
                + gamma * next_value_truncated
                - V_truncated[state]
            )
            weighted_truncated_error = truncated_rho * delta_truncated

            V_truncated[state] += alpha * weighted_truncated_error
            weighted_truncated[state].append(weighted_truncated_error)

    # Population variance; unvisited states remain exactly zero.
    var_standard = np.zeros(num_states, dtype=float)
    var_truncated = np.zeros(num_states, dtype=float)

    for state in range(num_states):
        if weighted_standard[state]:
            var_standard[state] = np.var(weighted_standard[state])

        if weighted_truncated[state]:
            var_truncated[state] = np.var(weighted_truncated[state])

    return {
        "V_standard": np.round(V_standard, 4),
        "V_truncated": np.round(V_truncated, 4),
        "var_standard": np.round(var_standard, 4),
        "var_truncated": np.round(var_truncated, 4)
    }