import numpy as np

def goal_conditioned_rollout(
    initial_state: np.ndarray,
    actions: np.ndarray,
    goal_state: np.ndarray,
    transition_matrix: np.ndarray,
    action_matrix: np.ndarray,
    goal_matrix: np.ndarray,
    anchor_strength: float = 0.5,
    anchor_power: float = 2.0
) -> dict:
    """
    Roll out a goal-conditioned world model with future anchoring.
    """
    initial_state = np.asarray(initial_state, dtype=float)
    actions = np.asarray(actions, dtype=float)
    goal_state = np.asarray(goal_state, dtype=float)
    transition_matrix = np.asarray(transition_matrix, dtype=float)
    action_matrix = np.asarray(action_matrix, dtype=float)
    goal_matrix = np.asarray(goal_matrix, dtype=float)

    T = len(actions)
    D = len(initial_state)

    trajectory = np.zeros((T + 1, D), dtype=float)
    raw_predictions = np.zeros((T, D), dtype=float)
    anchor_weights = np.zeros(T, dtype=float)

    trajectory[0] = initial_state

    for k in range(T):
        current_state = trajectory[k]

        # Goal-current difference.
        goal_diff = goal_state - current_state

        # Goal-conditioned model prediction.
        pre_activation = (
            transition_matrix @ current_state
            + action_matrix @ actions[k]
            + goal_matrix @ goal_diff
        )

        raw = np.tanh(pre_activation)
        raw_predictions[k] = raw

        # Future anchoring.
        alpha_k = anchor_strength * ((k + 1) / T) ** anchor_power
        anchor_weights[k] = alpha_k

        trajectory[k + 1] = (
            (1.0 - alpha_k) * raw
            + alpha_k * goal_state
        )

    return {
        "trajectory": trajectory,
        "anchor_weights": anchor_weights,
        "raw_predictions": raw_predictions
    }


def compute_future_anchor_loss(
    trajectory: np.ndarray,
    goal_state: np.ndarray,
    raw_predictions: np.ndarray,
    lambda_goal: float = 1.0,
    lambda_smooth: float = 0.1,
    lambda_consistency: float = 0.1
) -> dict:
    """
    Compute multi-component loss for the goal-conditioned world model.
    """
    trajectory = np.asarray(trajectory, dtype=float)
    goal_state = np.asarray(goal_state, dtype=float)
    raw_predictions = np.asarray(raw_predictions, dtype=float)

    T = raw_predictions.shape[0]

    # 1. Goal-reaching loss: final state vs. goal.
    goal_loss = np.sum((trajectory[-1] - goal_state) ** 2)

    # 2. Smoothness loss: average squared state changes.
    if T > 0:
        changes = trajectory[1:] - trajectory[:-1]
        smoothness_loss = np.mean(np.sum(changes ** 2, axis=1))
    else:
        smoothness_loss = 0.0

    # 3. Consistency loss: raw prediction vs. anchored prediction.
    if T > 0:
        consistency_error = trajectory[1:] - raw_predictions
        consistency_loss = np.mean(
            np.sum(consistency_error ** 2, axis=1)
        )
    else:
        consistency_loss = 0.0

    total_loss = (
        lambda_goal * goal_loss
        + lambda_smooth * smoothness_loss
        + lambda_consistency * consistency_loss
    )

    return {
        "goal_loss": float(goal_loss),
        "smoothness_loss": float(smoothness_loss),
        "consistency_loss": float(consistency_loss),
        "total_loss": float(total_loss)
    }