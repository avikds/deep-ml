import numpy as np


def option_value_iteration(
    R: np.ndarray,
    P: np.ndarray,
    options: list,
    gamma: float,
    num_iterations: int
) -> dict:
    """
    Compute value functions for options using iterative Bellman updates.
    """

    R = np.asarray(R, dtype=float)
    P = np.asarray(P, dtype=float)

    num_states, num_actions = R.shape
    num_options = len(options)

    # Q_options[s, o] = value of initiating option o in state s.
    Q_options = np.zeros((num_states, num_options), dtype=float)

    for _ in range(num_iterations):
        # ---------------------------------------------------------
        # 1. Derive V from the CURRENT option-value estimates.
        # ---------------------------------------------------------
        V = np.max(Q_options, axis=1)

        # ---------------------------------------------------------
        # 2. Compute upon-arrival values U(o, s).
        #
        # U(o, s) =
        #     (1-beta_o(s)) * Q_o(s)
        #     + beta_o(s) * V(s)
        # ---------------------------------------------------------
        U = np.zeros((num_options, num_states), dtype=float)

        for o, option in enumerate(options):
            termination = np.asarray(
                option["termination"],
                dtype=float
            )

            U[o] = (
                (1.0 - termination) * Q_options[:, o]
                + termination * V
            )

        # ---------------------------------------------------------
        # 3. Update every option-value using the CURRENT U.
        # ---------------------------------------------------------
        new_Q = np.zeros_like(Q_options)

        for o, option in enumerate(options):
            intra_policy = np.asarray(
                option["policy"],
                dtype=float
            )

            for s in range(num_states):
                option_value = 0.0

                for a in range(num_actions):
                    if intra_policy[s, a] == 0.0:
                        continue

                    # Expected immediate reward for this action.
                    reward = R[s, a]

                    # Expected upon-arrival value after transition.
                    future = np.dot(
                        P[s, a],
                        U[o]
                    )

                    option_value += intra_policy[s, a] * (
                        reward + gamma * future
                    )

                new_Q[s, o] = option_value

        Q_options = new_Q

    # -------------------------------------------------------------
    # Final V and U must reflect the state AFTER the final iteration.
    # -------------------------------------------------------------
    V = np.max(Q_options, axis=1)

    U = np.zeros((num_options, num_states), dtype=float)

    for o, option in enumerate(options):
        termination = np.asarray(
            option["termination"],
            dtype=float
        )

        U[o] = (
            (1.0 - termination) * Q_options[:, o]
            + termination * V
        )

    return {
        "Q_options": [
            [round(float(x), 4) for x in row]
            for row in Q_options
        ],
        "V": [round(float(x), 4) for x in V],
        "U": [
            [round(float(x), 4) for x in row]
            for row in U
        ]
    }