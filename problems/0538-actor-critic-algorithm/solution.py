import numpy as np


def actor_critic_step(
    state_features,
    action,
    reward,
    next_state_features,
    done,
    theta,
    w,
    gamma,
    alpha_theta,
    alpha_w
):
    """
    Perform one step of the Actor-Critic algorithm with linear function approximation.
    """

    state_features = np.asarray(state_features, dtype=float)
    next_state_features = np.asarray(next_state_features, dtype=float)
    theta = np.asarray(theta, dtype=float).copy()
    w = np.asarray(w, dtype=float).copy()

    # 1. Critic: compute current and next value estimates.
    v_current = np.dot(w, state_features)

    if done:
        v_next = 0.0
    else:
        v_next = np.dot(w, next_state_features)

    # TD error.
    td_error = float(
        reward + gamma * v_next - v_current
    )

    # 2. Critic update: semi-gradient TD(0).
    w_new = w + alpha_w * td_error * state_features

    # 3. Actor: compute probabilities using the OLD theta.
    logits = state_features @ theta
    logits = logits - np.max(logits)

    exp_logits = np.exp(logits)
    action_probs = exp_logits / np.sum(exp_logits)

    # 4. Score function for softmax policy.
    # grad log pi(a|s) = outer(phi(s), one_hot(a) - pi)
    one_hot = np.zeros(theta.shape[1], dtype=float)
    one_hot[action] = 1.0

    grad_log_policy = np.outer(
        state_features,
        one_hot - action_probs
    )

    # 5. Actor update using TD error as the advantage estimate.
    theta_new = theta + alpha_theta * td_error * grad_log_policy

    return (
        theta_new,
        w_new,
        td_error,
        action_probs
    )