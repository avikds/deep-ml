import numpy as np

def blocking_maze_dyna_q(
    rows: int,
    cols: int,
    initial_walls: list,
    added_walls: list,
    start: list,
    goal: list,
    change_step: int,
    gamma: float,
    alpha: float,
    epsilon: float,
    n_planning: int,
    n_steps: int,
    seed: int
) -> dict:
    """
    Simulate a Dyna-Q agent in a blocking maze where walls change mid-simulation.
    """

    np.random.seed(seed)

    n_states = rows * cols
    n_actions = 4

    Q = np.zeros((n_states, n_actions), dtype=float)

    walls = set(tuple(w) for w in initial_walls)
    added_walls = set(tuple(w) for w in added_walls)

    start_state = start[0] * cols + start[1]
    goal_state = goal[0] * cols + goal[1]

    # (state, action) -> (next_state, reward)
    model = {}

    def state_to_pos(state):
        return divmod(state, cols)

    def env_step(state, action):
        r, c = state_to_pos(state)

        dr = (-1, 1, 0, 0)[action]
        dc = (0, 0, -1, 1)[action]

        nr = r + dr
        nc = c + dc

        # Invalid moves leave the agent in place.
        if (
            nr < 0 or nr >= rows or
            nc < 0 or nc >= cols or
            (nr, nc) in walls
        ):
            next_state = state
        else:
            next_state = nr * cols + nc

        reward = 1.0 if next_state == goal_state else 0.0
        return next_state, reward

    def choose_action(state):
        if np.random.rand() < epsilon:
            return int(np.random.randint(n_actions))
        return int(np.argmax(Q[state]))

    current_state = start_state
    episodes_completed = 0
    cumulative_reward = 0.0

    for step in range(n_steps):
        # Apply the environmental change before acting at change_step.
        if step == change_step:
            walls.update(added_walls)

        action = choose_action(current_state)

        next_state, reward = env_step(current_state, action)
        done = (next_state == goal_state)

        cumulative_reward += reward

        # Real Q-learning update.
        if done:
            target = reward
        else:
            target = reward + gamma * np.max(Q[next_state])

        Q[current_state, action] += (
            alpha * (target - Q[current_state, action])
        )

        # Update the learned model with the real transition.
        model[(current_state, action)] = (next_state, reward)

        # Dyna-Q planning.
        model_keys = list(model.keys())

        for _ in range(n_planning):
            idx = np.random.randint(len(model_keys))
            s, a = model_keys[idx]
            s_next, r_model = model[(s, a)]

            # A transition into the goal is terminal.
            if s_next == goal_state:
                planning_target = r_model
            else:
                planning_target = r_model + gamma * np.max(Q[s_next])

            Q[s, a] += alpha * (planning_target - Q[s, a])

        if done:
            episodes_completed += 1
            current_state = start_state
        else:
            current_state = next_state

    # Count model entries that are inconsistent with the current environment.
    stale_entries = 0

    for (s, a), (stored_next, stored_reward) in model.items():
        actual_next, actual_reward = env_step(s, a)

        if (
            stored_next != actual_next
            or stored_reward != actual_reward
        ):
            stale_entries += 1

    return {
        "episodes_completed": episodes_completed,
        "cumulative_reward": round(float(cumulative_reward), 4),
        "model_entries": len(model),
        "stale_entries": stale_entries,
    }