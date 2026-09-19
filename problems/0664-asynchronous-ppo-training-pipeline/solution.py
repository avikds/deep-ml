import numpy as np

def async_ppo_pipeline(
    worker_batches: list,
    theta_init: list,
    alpha: float,
    gamma_stale: float,
    clip_eps: float,
    max_staleness: int
) -> dict:
    """
    Simulate asynchronous PPO training pipeline.
    """
    theta = np.asarray(theta_init, dtype=float).copy()

    learner_version = 0
    batches_used = 0
    batches_discarded = 0
    clip_fractions = []

    def softmax(logits):
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    for batch in worker_batches:
        policy_version = int(batch["policy_version"])
        lag = learner_version - policy_version

        # Discard stale batches.
        if lag > max_staleness:
            batches_discarded += 1
            continue

        states = np.asarray(batch["states"], dtype=float)
        actions = np.asarray(batch["actions"], dtype=int)
        old_log_probs = np.asarray(batch["old_log_probs"], dtype=float)
        advantages = np.asarray(batch["advantages"], dtype=float)

        batch_size = len(states)

        if batch_size == 0:
            clip_fractions.append(0.0)
            batches_used += 1
            learner_version += 1
            continue

        # Current policy probabilities and log-probabilities.
        logits = states @ theta
        probs = softmax(logits)

        new_log_probs = np.log(
            np.clip(probs[np.arange(batch_size), actions], 1e-8, 1.0)
        )

        # Importance sampling ratios.
        ratios = np.exp(new_log_probs - old_log_probs)

        surr1 = ratios * advantages
        clipped_ratios = np.clip(
            ratios,
            1.0 - clip_eps,
            1.0 + clip_eps
        )
        surr2 = clipped_ratios * advantages

        # Per the specification, a sample is clipped when surr1 > surr2.
        clipped = surr1 > surr2
        clip_fraction = float(np.mean(clipped))
        clip_fractions.append(clip_fraction)

        # Accumulate gradient only for non-clipped samples.
        grad = np.zeros_like(theta)

        for i in range(batch_size):
            if clipped[i]:
                continue

            # Softmax score:
            # d log pi(a|s) / d theta = outer(s, one_hot(a) - pi)
            score = -probs[i].copy()
            score[actions[i]] += 1.0

            grad += advantages[i] * ratios[i] * np.outer(
                states[i], score
            )

        # Average over all samples in the batch.
        grad /= batch_size

        # Staleness-decayed learning rate.
        effective_lr = alpha * (gamma_stale ** lag)

        theta += effective_lr * grad

        learner_version += 1
        batches_used += 1

    avg_clip_fraction = (
        float(np.mean(clip_fractions))
        if clip_fractions
        else 0.0
    )

    return {
        "theta": np.round(theta, 4).tolist(),
        "batches_used": int(batches_used),
        "batches_discarded": int(batches_discarded),
        "avg_clip_fraction": round(avg_clip_fraction, 4)
    }