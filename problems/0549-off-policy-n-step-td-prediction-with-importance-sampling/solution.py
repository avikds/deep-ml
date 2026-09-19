import numpy as np


def off_policy_nstep_td(
    episodes: list,
    behavior_policy: list,
    target_policy: list,
    num_states: int,
    num_actions: int,
    n: int,
    alpha: float,
    gamma: float
) -> np.ndarray:
    """
    Off-policy n-step TD prediction for state values using importance sampling.
    """

    V = np.zeros(num_states, dtype=float)

    behavior_policy = np.asarray(behavior_policy, dtype=float)
    target_policy = np.asarray(target_policy, dtype=float)

    if n <= 0:
        raise ValueError("n must be positive")

    for episode in episodes:
        T = len(episode)

        for t in range(T):
            state = int(episode[t][0])

            # -----------------------------------------------------
            # n-step return
            # -----------------------------------------------------
            end = min(t + n, T)

            G = 0.0

            # Accumulate rewards from t through end - 1.
            for k in range(t, end):
                reward = float(episode[k][2])
                G += (gamma ** (k - t)) * reward

            # If the n-step lookahead does not reach the end,
            # bootstrap from the value of the state after n steps.
            if end < T:
                next_state = int(episode[end][0])
                G += (gamma ** n) * V[next_state]

            # -----------------------------------------------------
            # Importance sampling correction
            # -----------------------------------------------------
            rho = 1.0

            # Correct for actions t through end - 1.
            for k in range(t, end):
                s_k = int(episode[k][0])
                a_k = int(episode[k][1])

                b_prob = behavior_policy[s_k, a_k]
                pi_prob = target_policy[s_k, a_k]

                if b_prob == 0.0:
                    rho = 0.0
                    break

                rho *= pi_prob / b_prob

            # -----------------------------------------------------
            # TD update
            # -----------------------------------------------------
            V[state] += alpha * rho * (G - V[state])

    return V