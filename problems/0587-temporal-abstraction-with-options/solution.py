import numpy as np


def options_value_iteration(
    n_states,
    terminal_states,
    transitions,
    options,
    gamma,
    theta=1e-10
):
    """
    Perform SMDP value iteration using temporally extended options.
    """

    terminal_states = set(terminal_states)

    V = np.zeros(n_states, dtype=float)
    policy = [0] * n_states

    # Precompute an option model:
    # R[s, o] = expected discounted reward until termination
    # F[s, o, s'] = discounted probability of terminating in s'
    option_rewards = []
    option_futures = []

    for option in options:
        initiation = set(option["initiation"])
        option_policy = option["policy"]
        termination = option["termination"]

        # States on which the option can actually continue.
        active_states = [
            s for s in range(n_states)
            if s not in terminal_states and s in initiation
            and s in option_policy
        ]

        active_index = {s: i for i, s in enumerate(active_states)}
        m = len(active_states)

        R_model = np.zeros(n_states, dtype=float)
        F_model = np.zeros((n_states, n_states), dtype=float)

        if m > 0:
            # A[x, y] = gamma * P(y | x, option) * (1-beta(y))
            A = np.zeros((m, m), dtype=float)

            # Immediate termination contribution to F.
            B = np.zeros((m, n_states), dtype=float)

            # Expected immediate reward.
            r_vec = np.zeros(m, dtype=float)

            for s in active_states:
                i = active_index[s]
                action = option_policy[s]

                if s not in transitions or action not in transitions[s]:
                    continue

                for next_state, prob, reward in transitions[s][action]:
                    prob = float(prob)
                    reward = float(reward)

                    r_vec[i] += prob * reward

                    # Terminal environment states terminate the option.
                    if next_state in terminal_states:
                        beta_next = 1.0
                    else:
                        beta_next = float(termination.get(next_state, 1.0))

                    # If the option terminates upon entering next_state,
                    # it contributes gamma * beta to the discounted
                    # termination-state distribution.
                    B[i, next_state] += gamma * prob * beta_next

                    # Otherwise the option continues.
                    if (
                        next_state in active_index
                        and beta_next < 1.0
                    ):
                        j = active_index[next_state]
                        A[i, j] += (
                            gamma
                            * prob
                            * (1.0 - beta_next)
                        )

            # R = r + A R
            R_active = np.linalg.solve(
                np.eye(m) - A,
                r_vec
            )

            # F = B + A F
            F_active = np.linalg.solve(
                np.eye(m) - A,
                B
            )

            for s, i in active_index.items():
                R_model[s] = R_active[i]
                F_model[s] = F_active[i]

        option_rewards.append(R_model)
        option_futures.append(F_model)

    # ---------------------------------------------------------
    # SMDP Value Iteration
    # ---------------------------------------------------------
    while True:
        delta = 0.0

        for s in range(n_states):
            if s in terminal_states:
                V[s] = 0.0
                policy[s] = 0
                continue

            best_value = -np.inf
            best_option = 0

            # Iterate in ascending option order so ties go to
            # the smallest option index.
            for o in range(len(options)):
                if s not in options[o]["initiation"]:
                    continue

                q = (
                    option_rewards[o][s]
                    + np.dot(option_futures[o][s], V)
                )

                if q > best_value:
                    best_value = q
                    best_option = o

            # If no option is available, keep value at zero.
            if best_value == -np.inf:
                best_value = 0.0
                best_option = 0

            delta = max(delta, abs(best_value - V[s]))
            V[s] = best_value
            policy[s] = best_option

        if delta < theta:
            break

    return (
        [round(float(v), 4) for v in V],
        policy
    )