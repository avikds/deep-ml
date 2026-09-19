import numpy as np
from typing import Optional


class MCTSNode:
    def __init__(self, state: int, parent: Optional['MCTSNode'] = None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0.0

    def is_leaf(self):
        return len(self.children) == 0

    def ucb1(self, c: float = 1.414) -> float:
        if self.visits == 0:
            return float("inf")

        if self.parent is None or self.parent.visits == 0:
            return self.value / self.visits

        return (
            self.value / self.visits
            + c * np.sqrt(
                np.log(self.parent.visits) / self.visits
            )
        )


def mcts_search(
    initial_state: int,
    max_value: int,
    iterations: int,
    seed: int = 42
) -> int:
    rng = np.random.RandomState(seed)

    root = MCTSNode(initial_state)

    def actions(state):
        return [
            a for a in (1, 2)
            if state + a <= max_value
        ]

    def terminal_value(state):
        if state == max_value:
            return 1.0
        if state > max_value:
            return -1.0
        return None

    for _ in range(iterations):
        node = root

        # Selection
        while True:
            result = terminal_value(node.state)
            if result is not None:
                break

            valid_actions = actions(node.state)

            # Expand an untried action first
            untried = [a for a in valid_actions if a not in node.children]

            if untried:
                action = untried[rng.randint(len(untried))]
                child = MCTSNode(
                    node.state + action,
                    parent=node
                )
                node.children[action] = child
                node = child
                break

            # Otherwise follow UCB1
            action, node = max(
                node.children.items(),
                key=lambda item: item[1].ucb1()
            )

        # Simulation
        result = terminal_value(node.state)

        if result is None:
            state = node.state
            player = 1  # player to move

            while True:
                valid_actions = actions(state)

                if not valid_actions:
                    # Going over the target loses for the player
                    # who made that move.
                    result = -1.0
                    break

                action = valid_actions[rng.randint(len(valid_actions))]
                state += action

                if state == max_value:
                    result = float(player)
                    break

                if state > max_value:
                    result = float(-player)
                    break

                player *= -1

        # Backpropagation.
        # node is always the current player at that node, while the
        # root wants the payoff from the root player's perspective.
        current = node
        payoff = result

        while current is not None:
            current.visits += 1
            current.value += payoff

            payoff = -payoff
            current = current.parent

    # Choose the most visited root action; break ties by larger action.
    if not root.children:
        return 1

    return max(
        root.children.items(),
        key=lambda item: (item[1].visits, item[0])
    )[0]