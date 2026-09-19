import numpy as np

# Implement your function below.
def sigmoid_moe_router(hidden_states, gate_weight, score_bias, top_k):
    """
    Implement sigmoid-based MoE routing with bias correction.

    Args:
        hidden_states (np.ndarray): Token representations, shape (num_tokens, hidden_dim).
        gate_weight (np.ndarray): Gate projection weights, shape (num_experts, hidden_dim).
        score_bias (np.ndarray): Learned bias for load balancing, shape (num_experts,).
        top_k (int): Number of experts to select per token.

    Returns:
        tuple: (top_k_weights, top_k_indices)
            - top_k_weights: Normalized routing weights, shape (num_tokens, top_k).
            - top_k_indices: Selected expert indices, shape (num_tokens, top_k).
    """
    # 1. Router logits
    # hidden_states: (num_tokens, hidden_dim)
    # gate_weight.T: (hidden_dim, num_experts)
    logits = hidden_states @ gate_weight.T

    # 2. Sigmoid routing scores (no softmax)
    sigmoid_scores = 1.0 / (1.0 + np.exp(-logits))

    # 3. Add bias only for expert selection
    selection_scores = sigmoid_scores + score_bias

    # 4. Select top-k experts for every token.
    # Stable sorting makes tie-breaking deterministic.
    top_k_indices = np.argsort(
        -selection_scores,
        axis=1,
        kind="stable"
    )[:, :top_k]

    # 5. Gather the ORIGINAL sigmoid weights, without bias.
    top_k_weights = np.take_along_axis(
        sigmoid_scores,
        top_k_indices,
        axis=1
    )

    # 6. Normalize selected weights so they sum to 1.
    weight_sum = np.sum(top_k_weights, axis=1, keepdims=True)
    top_k_weights = top_k_weights / weight_sum

    return top_k_weights, top_k_indices