import numpy as np

def anticipatory_moe(features, router_weights, expert_weights, delta):
    """
    Top-1 MoE forward pass with decoupled (anticipatory) routing.

    Args:
        features: list of T arrays, each of shape (N, d)
        router_weights: list of T arrays, each of shape (d, E)
        expert_weights: list of T arrays, each of shape (E, d, d_out)
        delta: non-negative integer, routing lookback offset

    Returns:
        List of T outputs, each a nested list of shape (N, d_out).
    """
    outputs = []

    for t in range(len(features)):
        x = np.asarray(features[t], dtype=float)

        # Use an earlier router snapshot when available.
        router_idx = t - delta if t >= delta else t
        W_r = np.asarray(router_weights[router_idx], dtype=float)

        # Top-1 routing.
        scores = x @ W_r
        expert_idx = np.argmax(scores, axis=1)

        # Apply the current step's expert weights.
        W_e = np.asarray(expert_weights[t], dtype=float)

        step_output = np.empty((x.shape[0], W_e.shape[2]), dtype=float)

        for i in range(x.shape[0]):
            step_output[i] = x[i] @ W_e[expert_idx[i]]

        outputs.append(step_output.tolist())

    return outputs