import numpy as np

def reinforce_baseline_update(
    episode: list,
    theta: np.ndarray,
    w: np.ndarray,
    gamma: float,
    alpha_theta: float,
    alpha_w: float
) -> dict:
    """
    Perform REINFORCE with baseline update for a single episode.

    Args:
        episode: list of (state, action, reward) tuples
        theta: policy parameters, shape (n_features, n_actions)
        w: baseline parameters, shape (n_features,)
        gamma: discount factor
        alpha_theta: learning rate for policy
        alpha_w: learning rate for baseline

    Returns:
        dict with 'theta', 'w', 'returns', 'advantages'
    """

    n_steps = len(episode)

    # Compute discounted returns G_t.
    returns = np.zeros(n_steps, dtype=float)
    G = 0.0

    for t in range(n_steps - 1, -1, -1):
        G = episode[t][2] + gamma * G
        returns[t] = G

    advantages = []

    # Process timesteps sequentially using the CURRENT parameters.
    for t, (state, action, _) in enumerate(episode):
        state = np.asarray(state, dtype=float)

        # Current baseline prediction.
        baseline = np.dot(w, state)

        # Advantage.
        advantage = returns[t] - baseline
        advantages.append(float(advantage))

        # Stable softmax of theta^T @ state.
        logits = state @ theta
        logits -= np.max(logits)

        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits)

        # Gradient of log pi(a | s).
        one_hot = np.zeros(theta.shape[1], dtype=float)
        one_hot[action] = 1.0

        grad_log_pi = np.outer(state, one_hot - probs)

        # REINFORCE policy update includes gamma^t.
        theta += (
            alpha_theta
            * (gamma ** t)
            * advantage
            * grad_log_pi
        )

        # Baseline update does NOT include gamma^t.
        w += alpha_w * advantage * state

    return {
        "theta": theta.tolist(),
        "w": w.tolist(),
        "returns": returns.tolist(),
        "advantages": advantages,
    }