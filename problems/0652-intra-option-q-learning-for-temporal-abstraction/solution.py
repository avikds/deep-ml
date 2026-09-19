import numpy as np

def intra_option_q_learning(
    num_states: int,
    num_actions: int,
    num_options: int,
    option_policies: list,
    option_terminations: list,
    initiation_sets: list,
    transitions: list,
    alpha: float,
    gamma: float
) -> list:
    """
    Learn option-value function Q(s, o) using intra-option Q-learning.
    """
    option_policies = np.asarray(option_policies, dtype=float)
    option_terminations = np.asarray(option_terminations, dtype=float)

    Q = np.zeros((num_states, num_options), dtype=float)

    # Membership lookup for fast initiation checks.
    initiation = [
        set(states) for states in initiation_sets
    ]

    for state, action, reward, next_state in transitions:
        for option in range(num_options):
            # Only update options whose policy can take this action.
            if option_policies[option, state, action] <= 0.0:
                continue

            q_current = Q[state, option]

            # Best available option at the next state.
            available = [
                o for o in range(num_options)
                if next_state in initiation[o]
            ]

            if available:
                best_option_value = max(
                    Q[next_state, o] for o in available
                )
            else:
                best_option_value = 0.0

            # U(s', o):
            # continue with probability (1-beta), terminate with beta.
            beta = option_terminations[option, next_state]

            continuation_value = (
                (1.0 - beta) * Q[next_state, option]
                + beta * best_option_value
            )

            target = reward + gamma * continuation_value

            Q[state, option] += alpha * (
                target - q_current
            )

    return np.round(Q, 4).tolist()