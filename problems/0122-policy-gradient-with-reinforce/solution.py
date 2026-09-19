import numpy as np

def compute_policy_gradient(
    theta: np.ndarray,
    episodes: list[list[tuple[int, int, float]]]
) -> np.ndarray:
    """
    Estimate the policy gradient using REINFORCE.
    """
    num_states, num_actions = theta.shape
    gradient = np.zeros_like(theta, dtype=float)

    if not episodes:
        return gradient

    for episode in episodes:
        G = 0.0

        # Compute returns and accumulate gradients backward
        for state, action, reward in reversed(episode):
            G += reward

            # Stable softmax
            logits = theta[state] - np.max(theta[state])
            exp_logits = np.exp(logits)
            probs = exp_logits / np.sum(exp_logits)

            # Gradient of log pi(a|s):
            # one_hot(action) - softmax(theta[state])
            grad_log_pi = -probs
            grad_log_pi[action] += 1.0

            gradient[state] += G * grad_log_pi

    # Average over episodes
    gradient /= len(episodes)

    return gradient