import numpy as np


def off_policy_nstep_sarsa(
    states: list,
    actions: list,
    rewards: list,
    Q: dict,
    target_policy: dict,
    behavior_policy: dict,
    n: int,
    gamma: float,
    alpha: float
) -> dict:
    """
    Perform off-policy n-step Sarsa with importance sampling on a single episode.
    """

    T = len(actions)

    if len(states) != T + 1 or len(rewards) != T:
        raise ValueError(
            "states must have length T+1 and rewards must have length T"
        )

    if n <= 0:
        raise ValueError("n must be positive")

    # Work on the supplied dictionary so its learned values are updated.
    Q = Q.copy()

    for t in range(T):
        # Horizon of the n-step return.
        h = min(t + n, T)

        # ---------------------------------------------------------
        # Compute n-step return:
        #
        # G = R[t] + gamma R[t+1] + ...
        #
        # If h < T, bootstrap from Q(S_h, A_h).
        # ---------------------------------------------------------
        G = 0.0

        for k in range(t, h):
            G += (gamma ** (k - t)) * rewards[k]

        if h < T:
            next_state = states[h]
            next_action = actions[h]

            G += (gamma ** (h - t)) * Q.get(
                (next_state, next_action),
                0.0
            )

        # ---------------------------------------------------------
        # Importance sampling ratio.
        #
        # Correct for actions AFTER (S_t, A_t) and BEFORE the
        # bootstrap pair:
        #
        # t+1, ..., h-1
        #
        # When there is no intermediate action, rho = 1.
        # ---------------------------------------------------------
        rho = 1.0

        for k in range(t + 1, h):
            s_k = states[k]
            a_k = actions[k]

            pi_prob = target_policy[s_k].get(a_k, 0.0)
            b_prob = behavior_policy[s_k].get(a_k, 0.0)

            if b_prob == 0.0:
                rho = 0.0
                break

            rho *= pi_prob / b_prob

        # ---------------------------------------------------------
        # Update Q(S_t, A_t) immediately.
        # ---------------------------------------------------------
        state_action = (states[t], actions[t])
        current_q = Q.get(state_action, 0.0)

        Q[state_action] = (
            current_q
            + alpha * rho * (G - current_q)
        )

    return Q