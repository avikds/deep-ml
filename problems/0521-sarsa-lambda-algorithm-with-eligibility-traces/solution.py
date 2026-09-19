import numpy as np


def sarsa_lambda(
    n_states: int,
    n_actions: int,
    transitions: dict,
    terminal_states: list,
    gamma: float,
    alpha: float,
    lam: float,
    epsilon: float,
    num_episodes: int,
    seed: int = 42
) -> list:
    """
    Sarsa(lambda) with accumulating eligibility traces.
    """

    rng = np.random.RandomState(seed)

    # Q(s, a) initialized to zero.
    Q = np.zeros((n_states, n_actions), dtype=float)

    terminal_set = set(terminal_states)
    non_terminal_states = [
        s for s in range(n_states)
        if s not in terminal_set
    ]

    def choose_action(state):
        # Explore uniformly.
        if rng.random() < epsilon:
            return int(rng.randint(n_actions))

        # Greedy action. np.argmax picks the smallest index on ties.
        return int(np.argmax(Q[state]))

    for _ in range(num_episodes):
        # Reset accumulating traces at the start of each episode.
        eligibility = np.zeros((n_states, n_actions), dtype=float)

        # Random non-terminal starting state, uniformly.
        state = int(rng.choice(non_terminal_states))
        action = choose_action(state)

        while state not in terminal_set:
            # Deterministic transition.
            if (state, action) not in transitions:
                break

            next_state, reward = transitions[(state, action)]
            next_state = int(next_state)
            reward = float(reward)

            # Accumulating trace for the current state-action pair.
            eligibility[state, action] += 1.0

            # Terminal states have Q = 0.
            if next_state in terminal_set:
                q_next = 0.0
            else:
                next_action = choose_action(next_state)
                q_next = Q[next_state, next_action]

            # SARSA TD error.
            delta = reward + gamma * q_next - Q[state, action]

            # Update every state-action pair using its eligibility.
            Q += alpha * delta * eligibility

            # Decay traces after the update.
            eligibility *= gamma * lam

            # End episode at terminal state.
            if next_state in terminal_set:
                break

            state = next_state
            action = next_action

    return [
        [round(float(x), 4) for x in row]
        for row in Q
    ]