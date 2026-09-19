import numpy as np

def jobshop_td_learning(jobs, n_machines, n_episodes, gamma, alpha, epsilon, seed=42):
    """
    Learn a scheduling policy for job-shop scheduling using SARSA (TD(0)).
    """
    rng = np.random.RandomState(seed)

    n_jobs = len(jobs)
    total_ops = sum(len(job) for job in jobs)

    q_values = {}

    def get_q(state, action):
        return q_values.get((state, action), 0.0)

    def valid_actions(state):
        return [
            j for j in range(n_jobs)
            if state[j] < len(jobs[j])
        ]

    def select_action(state):
        actions = valid_actions(state)

        # Epsilon exploration.
        if rng.random() < epsilon:
            return int(actions[rng.randint(len(actions))])

        # Greedy selection with lowest-index tie breaking.
        best_action = actions[0]
        best_q = get_q(state, best_action)

        for action in actions[1:]:
            q = get_q(state, action)
            if q > best_q:
                best_q = q
                best_action = action

        return int(best_action)

    best_makespan = float("inf")
    best_schedule = None

    initial_state = tuple([0] * n_jobs)

    for _ in range(n_episodes):
        state = initial_state

        machine_end = np.zeros(n_machines, dtype=float)
        job_end = np.zeros(n_jobs, dtype=float)

        schedule = []

        for _step in range(total_ops):
            action = select_action(state)

            op_idx = state[action]
            machine, duration = jobs[action][op_idx]

            start_time = max(
                machine_end[machine],
                job_end[action]
            )
            finish_time = start_time + duration

            machine_end[machine] = finish_time
            job_end[action] = finish_time

            next_state_list = list(state)
            next_state_list[action] += 1
            next_state = tuple(next_state_list)

            schedule.append(action)

            done = (len(schedule) == total_ops)

            if done:
                makespan = float(np.max(machine_end))
                reward = -makespan

                # Prefer smaller makespan; for ties, prefer
                # lexicographically smaller schedule.
                if (
                    makespan < best_makespan
                    or (
                        makespan == best_makespan
                        and (best_schedule is None or schedule < best_schedule)
                    )
                ):
                    best_makespan = makespan
                    best_schedule = schedule.copy()

                target = reward
            else:
                reward = 0.0
                next_action = select_action(next_state)
                target = reward + gamma * get_q(next_state, next_action)

            old_q = get_q(state, action)
            q_values[(state, action)] = (
                old_q + alpha * (target - old_q)
            )

            state = next_state

    return {
        "best_makespan": round(float(best_makespan), 4),
        "best_schedule": best_schedule,
        "q_values": {
            (state, action): round(float(value), 4)
            for (state, action), value in q_values.items()
        }
    }