def differential_sarsa(
    transitions: dict,
    initial_state: str,
    alpha: float,
    beta: float,
    num_steps: int
) -> tuple:
    """
    Differential Sarsa for the average-reward continuing setting.
    """

    # Initialize Q for every state-action pair in transitions.
    Q = {key: 0.0 for key in transitions}

    # Running estimate of average reward.
    R_bar = 0.0

    state = initial_state

    def greedy_action(state):
        # Find all available actions for this state.
        actions = [
            action for (s, action) in transitions
            if s == state
        ]

        if not actions:
            raise KeyError(f"No actions available for state {state}")

        # Lexicographically smallest action wins ties.
        return min(
            actions,
            key=lambda action: (-Q[(state, action)], action)
        )

    # Initial action selected greedily.
    action = greedy_action(state)

    for _ in range(num_steps):
        # Deterministic transition.
        reward, next_state = transitions[(state, action)]

        # Select the next action greedily.
        next_action = greedy_action(next_state)

        # Differential Sarsa TD error.
        delta = (
            reward
            - R_bar
            + Q[(next_state, next_action)]
            - Q[(state, action)]
        )

        # Update average reward.
        R_bar += beta * delta

        # Update current action-value.
        Q[(state, action)] += alpha * delta

        # Move to next state/action.
        state = next_state
        action = next_action

    return Q, R_bar