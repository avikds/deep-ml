import numpy as np


class Agent:
    """Online model-based agent for deterministic, changing gridworld mazes."""

    def __init__(self, n_states, n_actions, total_steps):
        self.n = n_states
        self.na = n_actions
        self.total_steps = total_steps
        self.t = 0

        # -1 means that this state-action transition has not been tested.
        self.trans = np.full((n_states, n_actions), -1, dtype=np.int32)

        self.goal = -1
        self.start = -1

        # Randomized action order avoids a fixed directional exploration bias.
        self.order = np.array([
            np.random.permutation(n_actions)
            for _ in range(n_states)
        ])

        # Times when transitions were last observed.
        self.last_edge = np.full(
            (n_states, n_actions), -1000000000, dtype=np.int64
        )
        self.last_state_probe = np.full(
            n_states, -1000000000, dtype=np.int64
        )

        # Retest a blocked direction at a visited route state periodically.
        self.probe_period = 250

    def _bfs(self, start):
        """Shortest paths using all currently known non-self transitions."""
        dist = np.full(self.n, -1, dtype=np.int32)
        first = np.full(self.n, -1, dtype=np.int32)

        queue = [int(start)]
        dist[start] = 0

        for s in queue:
            for a in range(self.na):
                ns = int(self.trans[s, a])

                if ns < 0 or ns == s or dist[ns] >= 0:
                    continue

                dist[ns] = dist[s] + 1
                first[ns] = a if s == start else first[s]
                queue.append(ns)

        return dist, first, queue

    def _unknown_action(self, state):
        """Choose an untested action at this state."""
        for a in self.order[state]:
            if self.trans[state, a] < 0:
                return int(a)
        return -1

    def _explore(self, state, dist, first, queue):
        """Move to the nearest reachable state with an untested action."""
        target = -1
        best_key = None

        for s in queue:
            missing = int(np.sum(self.trans[s] < 0))
            if missing == 0:
                continue

            # Prefer the nearest state; break ties in favor of more unknowns.
            key = (int(dist[s]), -missing)
            if best_key is None or key < best_key:
                best_key = key
                target = s

        if target < 0:
            return -1

        if target == state:
            return self._unknown_action(state)

        return int(first[target])

    def _probe_wall(self, state):
        """Retest one previously blocked direction at the current state."""
        blocked = [
            a for a in range(self.na)
            if self.trans[state, a] == state
        ]

        if not blocked:
            return -1

        # Try the direction that has gone the longest without being checked.
        a = min(blocked, key=lambda x: self.last_edge[state, x])
        return int(a)

    def _search_for_route(self, state, dist, first, queue):
        """
        If the goal is disconnected in the current model, search reachable
        blocked transitions. One may now be an opening.
        """
        target = -1
        target_action = -1
        best_key = None

        for s in queue:
            for a in range(self.na):
                if self.trans[s, a] != s:
                    continue

                # Prefer nearby transitions that have not been retested recently.
                key = (int(dist[s]), int(self.last_edge[s, a]))

                if best_key is None or key < best_key:
                    best_key = key
                    target = s
                    target_action = a

        if target < 0:
            return int(np.random.randint(self.na))

        if target == state:
            return int(target_action)

        return int(first[target])

    def act(self, state):
        state = int(state)

        if self.start < 0:
            self.start = state

        dist, first, queue = self._bfs(state)

        # Explore until all transitions in the discovered connected region
        # have been tested. This discovers the maze and the goal.
        action = self._explore(state, dist, first, queue)
        if action >= 0:
            return int(action)

        # The goal is known and the discovered map is fully explored.
        if self.goal >= 0 and dist[self.goal] >= 0:
            # Periodically retest a blocked direction at the current state.
            # An opening is discovered immediately if one has appeared.
            if self.t - self.last_state_probe[state] >= self.probe_period:
                action = self._probe_wall(state)
                if action >= 0:
                    return int(action)

            # Otherwise follow the shortest known route to the goal.
            action = int(first[self.goal])
            if action >= 0:
                return action

            return int(np.random.randint(self.na))

        # No known route: investigate blocked transitions in the reachable
        # component until a newly opened route is found.
        return self._search_for_route(state, dist, first, queue)

    def learn(self, state, action, reward, next_state, done):
        state = int(state)
        action = int(action)
        next_state = int(next_state)

        # Update the deterministic transition model, including changes.
        self.trans[state, action] = next_state
        self.last_edge[state, action] = self.t

        if next_state == state:
            self.last_state_probe[state] = self.t

        if reward > 0 or done:
            self.goal = next_state

        self.t += 1