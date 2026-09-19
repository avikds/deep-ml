import numpy as np


def q_lambda_watkins(
    episodes: list,
    n_states: int,
    n_actions: int,
    gamma: float,
    alpha: float,
    lam: float
) -> list:
    """
    Implement Watkins's Q(lambda) with eligibility traces.

    Args:
        episodes: List of episodes, each a list of
                  (state, action, reward, next_state, done) tuples
        n_states: Number of states
        n_actions: Number of actions
        gamma: Discount factor
        alpha: Learning rate
        lam: Lambda parameter for eligibility traces

    Returns:
        Q-value table as nested list, rounded to 4 decimal places.
    """

    # Initialize Q-values.
    Q = np.zeros((n_states, n_actions), dtype=float)

    for episode in episodes:
        # Reset eligibility traces for each episode.
        eligibility = np.zeros((n_states, n_actions), dtype=float)

        for t, (state, action, reward, next_state, done) in enumerate(episode):
            state = int(state)
            action = int(action)
            next_state = int(next_state)
            reward = float(reward)

            # Watkins Q-learning TD target.
            if done:
                q_next = 0.0
            else:
                q_next = np.max(Q[next_state])

            delta = reward + gamma * q_next - Q[state, action]

            # Accumulating eligibility trace.
            eligibility[state, action] += 1.0

            # Update all eligible state-action pairs.
            Q += alpha * delta * eligibility

            if done:
                # Episode terminates; no future action to inspect.
                break

            # The next behavior action is the action taken by the
            # pre-collected episode at the next timestep.
            if t + 1 < len(episode):
                next_action = int(episode[t + 1][1])

                # Greedy action under the UPDATED Q-table.
                greedy_action = int(np.argmax(Q[next_state]))

                if next_action == greedy_action:
                    # Continue traces for greedy behavior.
                    eligibility *= gamma * lam
                else:
                    # Watkins trace cutting for a non-greedy action.
                    eligibility.fill(0.0)
            else:
                # Defensive case for a truncated non-terminal episode.
                eligibility *= gamma * lam

    return [
        [round(float(x), 4) for x in row]
        for row in Q
    ]