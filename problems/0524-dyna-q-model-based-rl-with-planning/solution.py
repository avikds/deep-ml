import numpy as np


def dyna_q(
    num_states,
    num_actions,
    P,
    R,
    terminal_states,
    alpha,
    gamma,
    epsilon,
    num_episodes,
    n_planning
):
    """
    Implement the Dyna-Q algorithm.
    """

    P = np.asarray(P, dtype=float)
    R = np.asarray(R, dtype=float)

    Q = np.zeros((num_states, num_actions), dtype=float)

    terminal_set = set(terminal_states)

    # Model: (state, action) -> (reward, next_state)
    model = {}

    # States from which episodes may start.
    non_terminal_states = [
        s for s in range(num_states)
        if s not in terminal_set
    ]

    def choose_action(state):
        # Epsilon-greedy action selection.
        if np.random.random() < epsilon:
            return int(np.random.randint(num_actions))

        # np.argmax resolves ties by choosing the smallest index.
        return int(np.argmax(Q[state]))

    def sample_transition(state, action):
        """
        Sample next state from P[s, a, :].
        """
        probabilities = P[state, action]

        # np.random.choice expects probabilities summing to 1.
        return int(np.random.choice(num_states, p=probabilities))

    for _ in range(num_episodes):
        if not non_terminal_states:
            break

        # Random non-terminal starting state.
        state = int(np.random.choice(non_terminal_states))

        while state not in terminal_set:
            action = choose_action(state)

            # Real environment transition.
            next_state = sample_transition(state, action)
            reward = float(R[state, action])

            # -------------------------------------------------
            # 1. Direct RL: Q-learning update
            # -------------------------------------------------
            if next_state in terminal_set:
                target = reward
            else:
                target = reward + gamma * np.max(Q[next_state])

            Q[state, action] += alpha * (
                target - Q[state, action]
            )

            # -------------------------------------------------
            # 2. Model learning
            # -------------------------------------------------
            model[(state, action)] = (reward, next_state)

            # -------------------------------------------------
            # 3. Planning
            # -------------------------------------------------
            if model:
                observed_pairs = list(model.keys())

                for _ in range(n_planning):
                    # Uniformly sample a previously observed pair.
                    pair_idx = np.random.randint(len(observed_pairs))
                    sim_state, sim_action = observed_pairs[pair_idx]

                    sim_reward, sim_next_state = model[
                        (sim_state, sim_action)
                    ]

                    if sim_next_state in terminal_set:
                        sim_target = sim_reward
                    else:
                        sim_target = (
                            sim_reward
                            + gamma * np.max(Q[sim_next_state])
                        )

                    Q[sim_state, sim_action] += alpha * (
                        sim_target - Q[sim_state, sim_action]
                    )

            # Move to the next real state.
            if next_state in terminal_set:
                break

            state = next_state

    return Q