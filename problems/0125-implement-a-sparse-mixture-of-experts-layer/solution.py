import numpy as np

def moe(
    x: np.ndarray,
    We: np.ndarray,
    Wg: np.ndarray,
    n_experts: int,
    top_k: int
) -> np.ndarray:
    """
    Sparse Mixture-of-Experts layer with softmax gating and top-k routing.
    """
    batch, seq_len, d_model = x.shape

    top_k = min(top_k, n_experts)

    # Flatten tokens: (batch * seq_len, d_model)
    tokens = x.reshape(-1, d_model)

    # Gating logits and numerically stable softmax
    gate_logits = tokens @ Wg[:, :n_experts]
    gate_logits -= np.max(gate_logits, axis=1, keepdims=True)

    gate_probs = np.exp(gate_logits)
    gate_probs /= np.sum(gate_probs, axis=1, keepdims=True)

    # Indices of top-k experts
    top_idx = np.argpartition(
        gate_probs, -top_k, axis=1
    )[:, -top_k:]

    # Gather and renormalize selected gate probabilities
    selected = np.take_along_axis(gate_probs, top_idx, axis=1)
    selected /= np.sum(selected, axis=1, keepdims=True)

    # Aggregate expert outputs
    output = np.zeros_like(tokens, dtype=float)

    for j in range(top_k):
        experts = top_idx[:, j]
        weights = selected[:, j]

        # Process tokens grouped by expert
        for e in range(n_experts):
            mask = experts == e
            if np.any(mask):
                output[mask] += (
                    tokens[mask] @ We[e]
                ) * weights[mask, None]

    return output.reshape(batch, seq_len, d_model)