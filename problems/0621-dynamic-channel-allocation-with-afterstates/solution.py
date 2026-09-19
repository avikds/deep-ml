import numpy as np

def channel_afterstate_step(
    allocation: np.ndarray,
    call_cell: int,
    adjacency: np.ndarray,
    V: dict,
    alpha: float,
    gamma: float,
    prev_afterstate_key,
    prev_reward: float
) -> dict:
    """
    Perform one step of dynamic channel allocation using afterstate values.
    """
    allocation = np.asarray(allocation, dtype=int)

    n_cells, n_channels = allocation.shape

    # TD(0) update for the previous afterstate, using the current state's
    # eventual afterstate value. We perform this after selecting the current
    # afterstate below.
    candidates = []

    # Candidate accepting actions.
    for ch in range(n_channels):
        if allocation[call_cell, ch] != 0:
            continue

        new_allocation = allocation.copy()
        new_allocation[call_cell, ch] = 1

        # Count only newly created conflicts involving the requesting cell.
        additional_interference = 0
        for other in range(n_cells):
            if other == call_cell:
                continue

            if adjacency[call_cell, other] and allocation[other, ch] == 1:
                if allocation[call_cell, ch] == 0:
                    additional_interference += 1

        reward = 1.0 - additional_interference
        key = tuple(tuple(int(x) for x in row) for row in new_allocation)

        score = reward + gamma * V.get(key, 0.0)

        # Acceptance gets priority over rejection on ties, and lower
        # channel index gets priority among accepting actions.
        candidates.append((score, 1, -ch, ch, reward, key))

    # Rejection candidate.
    reject_key = tuple(tuple(int(x) for x in row) for row in allocation)
    reject_reward = 0.0
    reject_score = reject_reward + gamma * V.get(reject_key, 0.0)
    candidates.append((reject_score, 0, 0, -1, reject_reward, reject_key))

    # Maximize score; then prefer acceptance; then lowest channel index.
    best = max(candidates, key=lambda x: (x[0], x[1], x[2]))

    _, _, _, chosen_channel, reward, afterstate_key = best

    # TD update of the previous afterstate.
    if prev_afterstate_key is not None:
        current_value = V.get(afterstate_key, 0.0)
        prev_value = V.get(prev_afterstate_key, 0.0)

        V[prev_afterstate_key] = prev_value + alpha * (
            prev_reward + gamma * current_value - prev_value
        )

    return {
        "chosen_channel": int(chosen_channel),
        "reward": round(float(reward), 4),
        "afterstate_key": afterstate_key,
        "V": {
            key: round(float(value), 4)
            for key, value in V.items()
        }
    }