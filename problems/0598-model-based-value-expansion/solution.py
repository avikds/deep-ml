import numpy as np

def model_based_value_expansion(
    Q,
    transitions,
    rewards_model,
    V,
    experiences,
    H,
    gamma,
    alpha,
    terminal_states
):
    """
    Perform model-based value expansion updates on Q-values.
    """
    Q = np.asarray(Q, dtype=float).copy()
    V = np.asarray(V, dtype=float)
    terminal_states = set(terminal_states)

    for state, action in experiences:
        current_state = state
        current_action = action
        G = 0.0
        discount = 1.0
        terminated = False

        for step in range(H):
            # Simulate one model step.
            next_state = transitions[(current_state, current_action)]
            reward = rewards_model[(current_state, current_action)]

            G += discount * reward
            discount *= gamma

            # Stop immediately if the reached state is terminal.
            if next_state in terminal_states:
                terminated = True
                break

            current_state = next_state

            # Greedy action for the next model step.
            current_action = int(np.argmax(Q[current_state]))

        # Bootstrap only when the rollout did not terminate.
        if not terminated:
            G += discount * V[current_state]

        # Standard TD-style Q update.
        Q[state, action] += alpha * (G - Q[state, action])

    return np.round(Q, 4)