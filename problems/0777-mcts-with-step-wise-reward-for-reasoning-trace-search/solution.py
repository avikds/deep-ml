import math

def mcts_reasoning_trace(reward_table, branching, depth, n_simulations, c):
    """
    UCB1-MCTS over a B-ary tree of depth `depth` with a step-wise reward model.
    """

    class Node:
        def __init__(self, path):
            self.path = path
            self.children = {}
            self.visits = 0
            self.total = 0.0

    root = Node(())

    def step_reward(path):
        return float(reward_table.get(tuple(path), 0.0))

    def rollout(path):
        """Greedily complete the path using immediate step rewards."""
        path = list(path)

        while len(path) < depth:
            best_action = 0
            best_reward = step_reward(path + [0])

            for action in range(1, branching):
                reward = step_reward(path + [action])
                if reward > best_reward:
                    best_reward = reward
                    best_action = action

            path.append(best_action)

        return path

    best_trace = []
    best_value = -math.inf

    for _ in range(n_simulations):
        node = root
        selected_nodes = [root]

        # 1. Selection: descend while fully expanded.
        while len(node.path) < depth and len(node.children) == branching:
            best_action = None
            best_score = -math.inf

            for action in range(branching):
                child = node.children[action]

                # UCB1.
                q = child.total / child.visits
                exploration = c * math.sqrt(
                    math.log(node.visits) / child.visits
                )
                score = q + exploration

                # Smaller action index wins ties.
                if score > best_score:
                    best_score = score
                    best_action = action

            node = node.children[best_action]
            selected_nodes.append(node)

        # 2. Expansion: add the smallest untried action.
        if len(node.path) < depth:
            for action in range(branching):
                if action not in node.children:
                    child_path = node.path + (action,)
                    child = Node(child_path)
                    node.children[action] = child

                    node = child
                    selected_nodes.append(node)
                    break

        # 3. Rollout: greedily finish the trace.
        complete_trace = rollout(node.path)

        # Total reward of the complete trace.
        total_reward = sum(
            step_reward(complete_trace[:t])
            for t in range(1, depth + 1)
        )

        # Track best trace; strict > keeps earliest trace on ties.
        if total_reward > best_value:
            best_value = total_reward
            best_trace = complete_trace.copy()

        # 4. Backpropagation over selection + expansion path only.
        for visited_node in selected_nodes:
            visited_node.visits += 1
            visited_node.total += total_reward

    return best_trace