import numpy as np


def policy_iteration(
    num_states: int,
    num_actions: int,
    transitions: list,
    gamma: float,
    theta: float = 1e-8
) -> tuple:
    """
    Implement the Policy Iteration algorithm for solving MDPs.

    Args:
        num_states: Number of states in the MDP
        num_actions: Number of actions available in each state
        transitions: transitions[s][a] = [(prob, next_state, reward), ...]
        gamma: Discount factor
        theta: Convergence threshold for policy evaluation

    Returns:
        Tuple of (policy, values) where policy is a list of ints
        and values is a list of floats rounded to 4 decimal places
    """

    # Initial policy: action 0 for every state.
    policy = [0] * num_states

    # Initial value function.
    values = np.zeros(num_states, dtype=float)

    while True:
        # ------------------------------------------------------------
        # Policy Evaluation
        # ------------------------------------------------------------
        while True:
            delta = 0.0

            for s in range(num_states):
                action = policy[s]

                new_value = 0.0
                for prob, next_state, reward in transitions[s][action]:
                    new_value += prob * (reward + gamma * values[next_state])

                delta = max(delta, abs(new_value - values[s]))
                values[s] = new_value

            if delta < theta:
                break

        # ------------------------------------------------------------
        # Policy Improvement
        # ------------------------------------------------------------
        policy_stable = True

        for s in range(num_states):
            old_action = policy[s]

            q_values = np.zeros(num_actions, dtype=float)

            for a in range(num_actions):
                for prob, next_state, reward in transitions[s][a]:
                    q_values[a] += prob * (
                        reward + gamma * values[next_state]
                    )

            # np.argmax returns the first maximum, giving smallest
            # action index in the event of a tie.
            best_action = int(np.argmax(q_values))

            if best_action != old_action:
                policy[s] = best_action
                policy_stable = False

        # Stop once the policy no longer changes.
        if policy_stable:
            break

    return policy, [round(float(v), 4) for v in values]