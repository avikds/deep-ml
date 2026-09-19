import numpy as np


def off_policy_control_variate(
    rewards: list,
    values: list,
    target_probs: list,
    behavior_probs: list,
    dones: list,
    gamma: float
) -> tuple:
    """
    Compute off-policy return estimates using per-decision importance
    sampling with a control variate baseline.

    Args:
        rewards: List of T rewards
        values: List of T+1 state-value estimates (includes bootstrap)
        target_probs: List of T action probabilities under target policy
        behavior_probs: List of T action probabilities under behavior policy
        dones: List of T done flags (1=terminal, 0=non-terminal)
        gamma: Discount factor

    Returns:
        Tuple of (cv_returns, advantages) as lists rounded to 4 decimals
    """

    rewards = np.asarray(rewards, dtype=float)
    values = np.asarray(values, dtype=float)
    target_probs = np.asarray(target_probs, dtype=float)
    behavior_probs = np.asarray(behavior_probs, dtype=float)
    dones = np.asarray(dones, dtype=float)

    T = len(rewards)

    if len(values) != T + 1:
        raise ValueError("values must have length T + 1")

    if not (
        len(target_probs) == T
        and len(behavior_probs) == T
        and len(dones) == T
    ):
        raise ValueError("Per-step inputs must all have length T")

    cv_returns = np.zeros(T, dtype=float)

    # Start from the bootstrap value V(s_T).
    G = values[T]

    for t in range(T - 1, -1, -1):
        # Importance sampling ratio.
        if behavior_probs[t] == 0.0:
            rho = 0.0
        else:
            rho = target_probs[t] / behavior_probs[t]

        # Importance-sampled one-step target.
        next_value = 0.0 if dones[t] else G

        is_target = (
            rewards[t]
            + gamma * (1.0 - dones[t]) * next_value
        )

        # Control variate correction:
        # G_t = rho * target + (1-rho) * V_t
        G = rho * is_target + (1.0 - rho) * values[t]

        cv_returns[t] = G

    advantages = cv_returns - values[:T]

    return (
        [round(float(x), 4) for x in cv_returns],
        [round(float(x), 4) for x in advantages]
    )