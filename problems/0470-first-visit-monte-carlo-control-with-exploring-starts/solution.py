import numpy as np

def mc_control_exploring_starts(
    env: dict,
    n_states: int,
    n_actions: int,
    gamma: float,
    n_episodes: int,
    max_steps: int = 100,
    seed: int = 42
) -> tuple:
    """
    First-visit Monte Carlo control with exploring starts.
    """

    # Reproducibility
    np.random.seed(seed)

    # Q-values start at zero
    Q = np.zeros((n_states, n_actions), dtype=float)

    # Random initial policy
    policy = np.random.randint(0, n_actions, size=n_states)

    # Running sum and count of returns
    returns_sum = np.zeros((n_states, n_actions), dtype=float)
    returns_count = np.zeros((n_states, n_actions), dtype=int)

    for _ in range(n_episodes):
        episode = []

        # Exploring start: random initial state and action
        state = np.random.randint(0, n_states)
        action = np.random.randint(0, n_actions)

        for _ in range(max_steps):
            # Missing transition ends the episode immediately
            if (state, action) not in env:
                break

            next_state, reward, done = env[(state, action)]

            # Store (state, action, reward)
            episode.append((state, action, reward))

            if done:
                break

            # After the first step, follow the current policy
            state = next_state
            action = policy[state]

        # ---------------------------------------------------------
        # First-visit MC update
        # ---------------------------------------------------------
        G = 0.0
        visited = set()

        # Work backwards to efficiently calculate returns
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]

            G = reward + gamma * G

            # Update only the first occurrence in the episode
            if (state, action) not in visited:
                visited.add((state, action))

                returns_sum[state, action] += G
                returns_count[state, action] += 1

                Q[state, action] = (
                    returns_sum[state, action] /
                    returns_count[state, action]
                )

        # Greedy policy improvement
        for state in range(n_states):
            policy[state] = np.argmax(Q[state])

    return Q, policy