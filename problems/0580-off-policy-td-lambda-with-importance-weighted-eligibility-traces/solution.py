import numpy as np


def off_policy_td_lambda(
    episodes: list,
    behavior_policy: list,
    target_policy: list,
    num_states: int,
    num_actions: int,
    gamma: float,
    lam: float,
    alpha: float
) -> np.ndarray:
    """
    Off-policy TD(lambda) prediction with importance-weighted eligibility traces.
    """

    V = np.zeros(num_states, dtype=float)

    behavior_policy = np.asarray(behavior_policy, dtype=float)
    target_policy = np.asarray(target_policy, dtype=float)

    for episode in episodes:
        # Reset eligibility traces for every episode.
        e = np.zeros(num_states, dtype=float)

        for t, (state, action, reward) in enumerate(episode):
            state = int(state)
            action = int(action)
            reward = float(reward)

            # -----------------------------------------------------
            # Importance sampling ratio for current state-action.
            # -----------------------------------------------------
            b_prob = behavior_policy[state, action]
            pi_prob = target_policy[state, action]

            if b_prob == 0.0:
                rho = 0.0
            else:
                rho = pi_prob / b_prob

            # -----------------------------------------------------
            # Next-state value.
            #
            # The episode terminates after its last tuple, so the
            # final transition has no bootstrap value.
            # -----------------------------------------------------
            if t == len(episode) - 1:
                v_next = 0.0
            else:
                next_state = int(episode[t + 1][0])
                v_next = V[next_state]

            # TD error.
            delta = reward + gamma * v_next - V[state]

            # -----------------------------------------------------
            # Importance-weighted eligibility trace:
            #
            # 1. decay existing traces
            # 2. add indicator for current state
            # 3. multiply all traces by rho
            # -----------------------------------------------------
            e *= gamma * lam
            e[state] += 1.0
            e *= rho

            # Update every state's value.
            V += alpha * delta * e

    return V