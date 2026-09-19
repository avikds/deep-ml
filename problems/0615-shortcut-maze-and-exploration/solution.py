import numpy as np

def shortcut_maze_exploration(
    grid_rows: int,
    grid_cols: int,
    walls_phase1: list,
    walls_phase2: list,
    change_step: int,
    start: list,
    goal: tuple,
    total_steps: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    kappa: float,
    n_planning: int,
    seed: int
) -> dict:
    """
    Simulate a Dyna-Q+ agent in a changing maze environment.
    """

    rng = np.random.RandomState(seed)

    n_states = grid_rows * grid_cols
    n_actions = 4

    # Q-table.
    Q = np.zeros((n_states, n_actions), dtype=float)

    # Model: (state, action) -> (next_state, reward)
    model = {}

    # Last real-environment visit timestep for each stored pair.
    last_visit = {}

    phase1_walls = set(tuple(x) for x in walls_phase1)
    phase2_walls = set(tuple(x) for x in walls_phase2)
    goal = tuple(goal)

    start_state = start[0] * grid_cols + start[1]
    current_state = start_state

    episodes_completed = 0
    total_reward = 0.0
    cumulative_reward = []

    def step_environment(state, action, walls):
        r = state // grid_cols
        c = state % grid_cols

        if action == 0:      # up
            nr, nc = r - 1, c
        elif action == 1:    # down
            nr, nc = r + 1, c
        elif action == 2:    # left
            nr, nc = r, c - 1
        else:                # right
            nr, nc = r, c + 1

        if (
            nr < 0 or nr >= grid_rows or
            nc < 0 or nc >= grid_cols or
            (nr, nc) in walls
        ):
            nr, nc = r, c

        next_state = nr * grid_cols + nc
        reward = 1.0 if (nr, nc) == goal else 0.0

        return next_state, reward

    def select_action(state):
        # Epsilon exploration.
        if rng.random() < epsilon:
            return int(rng.randint(n_actions))

        # Random tie-breaking among greedy actions.
        q_values = Q[state]
        max_q = np.max(q_values)
        best_actions = np.flatnonzero(q_values == max_q)
        return int(best_actions[rng.randint(len(best_actions))])

    for step in range(total_steps):
        walls = phase1_walls if step < change_step else phase2_walls

        # 1. Epsilon-greedy action selection.
        action = select_action(current_state)

        # 2. Real environment transition.
        next_state, reward = step_environment(
            current_state, action, walls
        )

        # 3. Real Q-learning update.
        target = reward + gamma * np.max(Q[next_state])
        Q[current_state, action] += alpha * (
            target - Q[current_state, action]
        )

        # 4. Store/update model and real visit time.
        model[(current_state, action)] = (next_state, reward)
        last_visit[(current_state, action)] = step

        # 5. Dyna-Q+ planning.
        model_keys = list(model.keys())

        for _ in range(n_planning):
            idx = rng.randint(len(model_keys))
            s, a = model_keys[idx]

            s_next, r_model = model[(s, a)]

            elapsed = step - last_visit[(s, a)]
            bonus = kappa * np.sqrt(elapsed)

            planning_target = (
                r_model
                + bonus
                + gamma * np.max(Q[s_next])
            )

            Q[s, a] += alpha * (
                planning_target - Q[s, a]
            )

        # 6. Episode completion/reset.
        if next_state == goal[0] * grid_cols + goal[1]:
            episodes_completed += 1
            total_reward += reward
            current_state = start_state
        else:
            total_reward += reward
            current_state = next_state

        cumulative_reward.append(float(total_reward))

    return {
        "episodes_completed": int(episodes_completed),
        "total_reward": round(float(total_reward), 4),
        "cumulative_reward": cumulative_reward
    }