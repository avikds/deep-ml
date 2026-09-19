import numpy as np


def distributed_onpolicy_step(
    worker_data: list,
    gamma: float,
    lam: float,
    clip_eps: float
) -> dict:
    """
    Simulate one optimization step of distributed on-policy RL.
    """

    all_advantages = []
    all_returns = []
    all_values = []
    all_old_log_probs = []
    all_new_log_probs = []
    per_worker_mean_return = []

    # ---------------------------------------------------------
    # Compute GAE independently for each worker.
    # ---------------------------------------------------------
    for worker in worker_data:
        rewards = np.asarray(worker["rewards"], dtype=float)
        values = np.asarray(worker["values"], dtype=float)
        dones = np.asarray(worker["dones"], dtype=float)

        old_log_probs = np.asarray(
            worker["old_log_probs"], dtype=float
        )
        new_log_probs = np.asarray(
            worker["new_log_probs"], dtype=float
        )

        T = len(rewards)

        advantages = np.zeros(T, dtype=float)

        gae = 0.0

        # Compute GAE backwards.
        for t in range(T - 1, -1, -1):
            # No bootstrap across a terminal transition.
            nonterminal = 1.0 - dones[t]

            delta = (
                rewards[t]
                + gamma * values[t + 1] * nonterminal
                - values[t]
            )

            gae = delta + gamma * lam * nonterminal * gae
            advantages[t] = gae

        # GAE return estimate.
        returns = advantages + values[:T]

        all_advantages.append(advantages)
        all_returns.append(returns)
        all_values.append(values[:T])
        all_old_log_probs.append(old_log_probs)
        all_new_log_probs.append(new_log_probs)

        per_worker_mean_return.append(
            float(np.mean(returns)) if T > 0 else 0.0
        )

    # ---------------------------------------------------------
    # Pool all workers into one global batch.
    # ---------------------------------------------------------
    if all_advantages:
        advantages = np.concatenate(all_advantages)
        returns = np.concatenate(all_returns)
        values = np.concatenate(all_values)
        old_log_probs = np.concatenate(all_old_log_probs)
        new_log_probs = np.concatenate(all_new_log_probs)
    else:
        advantages = np.array([], dtype=float)
        returns = np.array([], dtype=float)
        values = np.array([], dtype=float)
        old_log_probs = np.array([], dtype=float)
        new_log_probs = np.array([], dtype=float)

    num_samples = len(advantages)

    # Raw advantage statistics.
    mean_advantage_raw = (
        float(np.mean(advantages)) if num_samples > 0 else 0.0
    )
    std_advantage_raw = (
        float(np.std(advantages)) if num_samples > 0 else 0.0
    )

    # ---------------------------------------------------------
    # Normalize pooled advantages.
    # ---------------------------------------------------------
    if num_samples > 0:
        if std_advantage_raw >= 1e-8:
            normalized_advantages = (
                advantages - mean_advantage_raw
            ) / std_advantage_raw
        else:
            normalized_advantages = advantages - mean_advantage_raw
    else:
        normalized_advantages = np.array([], dtype=float)

    # ---------------------------------------------------------
    # PPO clipped surrogate objective.
    # ---------------------------------------------------------
    if num_samples > 0:
        # Ratio = pi_new / pi_old
        ratios = np.exp(new_log_probs - old_log_probs)

        clipped_ratios = np.clip(
            ratios,
            1.0 - clip_eps,
            1.0 + clip_eps
        )

        surr1 = ratios * normalized_advantages
        surr2 = clipped_ratios * normalized_advantages

        # PPO minimizes negative surrogate objective.
        policy_loss = -float(np.mean(np.minimum(surr1, surr2)))

        # Fraction of samples whose ratio was actually clipped.
        clipped = (
            (ratios < 1.0 - clip_eps)
            | (ratios > 1.0 + clip_eps)
        )
        clip_fraction = float(np.mean(clipped))

        # Value MSE.
        value_loss = float(np.mean((returns - values) ** 2))
    else:
        policy_loss = 0.0
        value_loss = 0.0
        clip_fraction = 0.0

    return {
        "policy_loss": round(policy_loss, 4),
        "value_loss": round(value_loss, 4),
        "clip_fraction": round(clip_fraction, 4),
        "mean_advantage_raw": round(mean_advantage_raw, 4),
        "std_advantage_raw": round(std_advantage_raw, 4),
        "num_samples": int(num_samples),
        "per_worker_mean_return": [
            round(x, 4) for x in per_worker_mean_return
        ],
    }