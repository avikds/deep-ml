import numpy as np

def mcts(
    root_state,
    num_actions,
    get_legal_actions,
    apply_action,
    check_terminal,
    policy_value_fn,
    c_puct,
    num_simulations
):
    class Node:
        def __init__(self, state):
            self.state = state
            self.visits = 0
            self.priors = np.zeros(num_actions, dtype=float)
            self.legal_actions = []
            self.children = {}
            self.edge_visits = {}
            self.edge_total = {}
            self.terminal = False
            self.terminal_value = 0.0

    def expand(node):
        done, value = check_terminal(node.state)

        if done:
            node.terminal = True
            node.terminal_value = float(value)
            return float(value)

        node.legal_actions = sorted(
            int(a) for a in get_legal_actions(node.state)
        )

        if not node.legal_actions:
            return 0.0

        priors, value = policy_value_fn(node.state)
        priors = np.asarray(priors, dtype=float)

        masked = np.zeros(num_actions, dtype=float)
        for a in node.legal_actions:
            masked[a] = max(0.0, priors[a])

        total = masked.sum()

        if total > 0.0:
            masked /= total
        else:
            uniform = 1.0 / len(node.legal_actions)
            for a in node.legal_actions:
                masked[a] = uniform

        node.priors = masked
        return float(value)

    done, _ = check_terminal(root_state)
    if done:
        return [0.0] * num_actions

    root = Node(root_state)
    expand(root)

    for _ in range(num_simulations):
        node = root
        path = []

        # Selection + expansion.
        while True:
            if node.terminal:
                leaf_value = node.terminal_value
                break

            untried = [
                a for a in node.legal_actions
                if a not in node.children
            ]

            if untried:
                # Smallest untried action.
                action = untried[0]
                child = Node(apply_action(node.state, action))

                path.append((node, action))

                child_value = expand(child)
                node.children[action] = child
                node.edge_visits[action] = 0
                node.edge_total[action] = 0.0

                leaf_value = (
                    child.terminal_value
                    if child.terminal
                    else child_value
                )
                break

            # Fully expanded: PUCT selection.
            sqrt_n = np.sqrt(node.visits)
            best_action = None
            best_score = -np.inf

            for action in node.legal_actions:
                n_a = node.edge_visits[action]
                q_a = (
                    node.edge_total[action] / n_a
                    if n_a > 0 else 0.0
                )
                p_a = node.priors[action]

                score = q_a + c_puct * p_a * sqrt_n / (1.0 + n_a)

                if (
                    score > best_score
                    or (
                        score == best_score
                        and (
                            best_action is None
                            or action < best_action
                        )
                    )
                ):
                    best_score = score
                    best_action = action

            path.append((node, best_action))
            node = node.children[best_action]

        # Backup.
        value = leaf_value

        for parent, action in reversed(path):
            value = -value

            parent.edge_visits[action] += 1
            parent.edge_total[action] += value

        # Every simulation passes through the root.
        root.visits += 1

    visits = np.zeros(num_actions, dtype=float)

    for action in root.legal_actions:
        visits[action] = root.edge_visits.get(action, 0)

    total = visits.sum()

    if total == 0:
        return [0.0] * num_actions

    return (visits / total).tolist()