import numpy as np


def reinforce_with_baseline(episode, theta, w, gamma, alpha_theta, alpha_w):
    """
    Perform one episode update of REINFORCE with a value function baseline.
    
    Args:
        episode: list of (state, action, reward) tuples
        theta: np.ndarray of shape (n_states, n_actions), policy parameters
        w: np.ndarray of shape (n_states,), value function parameters
        gamma: float, discount factor
        alpha_theta: float, policy learning rate
        alpha_w: float, value function learning rate
    
    Returns:
        tuple: (theta_new, w_new) updated parameters
    """
    theta = np.asarray(theta, dtype=float).copy()
    w = np.asarray(w, dtype=float).copy()

    T = len(episode)

    # Compute discounted returns G_t.
    returns = np.zeros(T, dtype=float)
    G = 0.0

    for t in range(T - 1, -1, -1):
        _, _, reward = episode[t]
        G = float(reward) + gamma * G
        returns[t] = G

    # Process the episode sequentially.
    for t, (state, action, _) in enumerate(episode):
        state = int(state)
        action = int(action)

        # Stable softmax using the CURRENT policy parameters.
        logits = theta[state] - np.max(theta[state])
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits)

        # Current value baseline.
        value = w[state]

        # Advantage.
        advantage = returns[t] - value

        # Value-function update.
        w[state] += alpha_w * advantage

        # Gradient of log pi(a|s):
        # e_action - pi
        grad_log_pi = -probs.copy()
        grad_log_pi[action] += 1.0

        # Policy update, scaled by gamma^t.
        theta[state] += (
            alpha_theta
            * (gamma ** t)
            * advantage
            * grad_log_pi
        )

    return theta, w