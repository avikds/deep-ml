import numpy as np
import heapq


def prioritized_sweeping(
    num_states: int,
    num_actions: int,
    experiences: list,
    alpha: float,
    gamma: float,
    theta: float,
    n_planning: int
) -> np.ndarray:
    """
    Perform prioritized sweeping on a sequence of experiences.
    """

    # Q-table initialized to zero.
    Q = np.zeros((num_states, num_actions), dtype=float)

    # Learned model:
    # (state, action) -> (reward, next_state, done)
    model = {}

    # predecessors[s] = set of (state, action) pairs whose
    # learned transition leads to s.
    predecessors = [set() for _ in range(num_states)]

    # Max-heap implemented with negative priorities.
    priority_queue = []

    # Current best queued priority for each state-action pair.
    entry_finder = {}

    def push_priority(state, action, priority):
        """Insert/update a pair, keeping the maximum priority."""
        key = (state, action)

        if priority <= theta:
            return

        old_priority = entry_finder.get(key)

        if old_priority is None or priority > old_priority:
            entry_finder[key] = priority
            heapq.heappush(
                priority_queue,
                (-priority, state, action)
            )

    def pop_priority():
        """Pop the currently highest-priority valid pair."""
        while priority_queue:
            neg_priority, state, action = heapq.heappop(priority_queue)
            priority = -neg_priority
            key = (state, action)

            # Skip stale heap entries.
            if key not in entry_finder:
                continue

            if not np.isclose(priority, entry_finder[key]):
                continue

            del entry_finder[key]
            return state, action

        return None

    def q_target(state, action):
        """Bellman target using the learned model."""
        reward, next_state, done = model[(state, action)]

        if done:
            return float(reward)

        return float(reward + gamma * np.max(Q[next_state]))

    for state, action, reward, next_state, done in experiences:
        state = int(state)
        action = int(action)
        next_state = int(next_state)
        reward = float(reward)
        done = bool(done)

        key = (state, action)

        # ---------------------------------------------------------
        # 1. Update the learned model.
        # ---------------------------------------------------------

        # If this pair previously pointed to another state, remove
        # the obsolete predecessor relationship.
        if key in model:
            _, old_next_state, _ = model[key]
            predecessors[old_next_state].discard(key)

        model[key] = (reward, next_state, done)
        predecessors[next_state].add(key)

        # ---------------------------------------------------------
        # 2. Compute real-experience priority.
        # ---------------------------------------------------------
        target = q_target(state, action)
        priority = abs(target - Q[state, action])

        push_priority(state, action, priority)

        # ---------------------------------------------------------
        # 3. Prioritized planning.
        # ---------------------------------------------------------
        for _ in range(n_planning):
            popped = pop_priority()

            if popped is None:
                break

            s, a = popped

            # Update Q-value from the learned model.
            target = q_target(s, a)
            Q[s, a] += alpha * (target - Q[s, a])

            # After Q[s,a] changes, all predecessors of s
            # may now have significant TD errors.
            for pred_s, pred_a in predecessors[s]:
                pred_target = q_target(pred_s, pred_a)

                pred_priority = abs(
                    pred_target - Q[pred_s, pred_a]
                )

                push_priority(
                    pred_s,
                    pred_a,
                    pred_priority
                )

    return Q