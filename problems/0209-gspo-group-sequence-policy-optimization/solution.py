import numpy as np

def gspo_objective(
    log_probs_new: list[list[float]],
    log_probs_old: list[list[float]],
    rewards: list[float],
    epsilon: float = 0.2
) -> float:
    """
    Compute the GSPO sequence-level clipped objective.
    """

    rewards = np.asarray(rewards, dtype=float)

    # Group-relative advantages
    mean_r = np.mean(rewards)
    std_r = np.std(rewards)

    if std_r < 1e-12:
        advantages = rewards - mean_r
    else:
        advantages = (rewards - mean_r) / std_r

    objectives = []

    for i in range(len(rewards)):
        new_seq = np.asarray(log_probs_new[i], dtype=float)
        old_seq = np.asarray(log_probs_old[i], dtype=float)

        if len(new_seq) != len(old_seq):
            raise ValueError("Each new/old log-probability sequence must have the same length")

        if len(new_seq) == 0:
            raise ValueError("Log-probability sequences must not be empty")

        # Length-normalized sequence importance ratio
        log_ratio = np.sum(new_seq - old_seq) / len(new_seq)
        ratio = np.exp(log_ratio)

        clipped = np.clip(
            ratio,
            1.0 - epsilon,
            1.0 + epsilon
        )

        # GSPO clipped surrogate
        a = advantages[i]
        objective = min(
            ratio * a,
            clipped * a
        )

        objectives.append(objective)

    return float(np.mean(objectives))