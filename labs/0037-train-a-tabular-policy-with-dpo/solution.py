import numpy as np

def train(train_chosen, train_rejected, n_items, beta=0.5):
    """
    Train a tabular preference policy with DPO.
    """
    chosen = np.asarray(train_chosen, dtype=int)
    rejected = np.asarray(train_rejected, dtype=int)

    if len(chosen) != len(rejected):
        raise ValueError("train_chosen and train_rejected must have the same length")

    if n_items <= 0:
        raise ValueError("n_items must be positive")

    # Tabular policy logits. Since the reference policy is uniform,
    # its log-probability cancels from each pairwise difference.
    logits = np.zeros(n_items, dtype=np.float64)

    # A few hundred vectorized gradient steps.
    steps = 500
    lr = 0.05

    for _ in range(steps):
        diff = logits[chosen] - logits[rejected]

        # z = beta * (log pi(chosen) - log pi(rejected))
        # Uniform reference cancels and log-softmax normalization cancels.
        z = np.clip(beta * diff, -50.0, 50.0)

        # For loss = -log sigmoid(z),
        # dL/dz = sigmoid(z) - 1 = -sigmoid(-z)
        # Compute stably.
        grad_z = -1.0 / (1.0 + np.exp(z))

        grad_logits = np.zeros(n_items, dtype=np.float64)

        np.add.at(
            grad_logits,
            chosen,
            beta * grad_z
        )
        np.add.at(
            grad_logits,
            rejected,
            -beta * grad_z
        )

        # Average gradient over the training batch.
        grad_logits /= max(len(chosen), 1)

        logits -= lr * grad_logits

        # Remove irrelevant global shift for numerical stability.
        logits -= np.mean(logits)

    def score(indices):
        indices = np.asarray(indices, dtype=int)
        if np.any(indices < 0) or np.any(indices >= n_items):
            raise ValueError("indices out of range")
        return logits[indices].astype(float)

    return score