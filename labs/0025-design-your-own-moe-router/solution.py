import numpy as np

def route(x, W_gate):
    """
    Sparse top-2 MoE routing with numerically stable softmax.
    """
    logits = np.asarray(x, dtype=np.float64) @ np.asarray(W_gate, dtype=np.float64)

    # Numerical stabilization
    logits = logits - np.max(logits, axis=1, keepdims=True)

    # Full softmax
    exp_logits = np.exp(np.clip(logits, -50.0, 50.0))
    probs = exp_logits / np.maximum(
        np.sum(exp_logits, axis=1, keepdims=True), 1e-12
    )

    # Keep only the top-2 experts for each sample
    k = min(2, num_experts := probs.shape[1])
    top_idx = np.argpartition(probs, -k, axis=1)[:, -k:]

    weights = np.zeros_like(probs)

    rows = np.arange(probs.shape[0])[:, None]
    top_probs = probs[rows, top_idx]

    # Renormalize the selected experts
    top_probs /= np.maximum(
        np.sum(top_probs, axis=1, keepdims=True), 1e-12
    )

    weights[rows, top_idx] = top_probs

    # Final safety normalization
    weights /= np.maximum(
        np.sum(weights, axis=1, keepdims=True), 1e-12
    )

    return weights.astype(np.float64)