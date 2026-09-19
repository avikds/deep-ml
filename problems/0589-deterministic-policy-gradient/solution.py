import numpy as np

def deterministic_policy_gradient(
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    next_states: np.ndarray,
    dones: np.ndarray,
    theta: np.ndarray,
    w: np.ndarray,
    gamma: float,
    alpha_theta: float,
    alpha_w: float
) -> tuple:
    """
    Perform one update step of the Deterministic Policy Gradient algorithm
    with linear function approximation for both the actor and critic.
    """
    states = np.asarray(states, dtype=float)
    actions = np.asarray(actions, dtype=float)
    rewards = np.asarray(rewards, dtype=float)
    next_states = np.asarray(next_states, dtype=float)
    dones = np.asarray(dones, dtype=float)
    theta = np.asarray(theta, dtype=float)
    w = np.asarray(w, dtype=float)

    # Save the original critic parameters for the actor update.
    w_old = w.copy()

    # ----- Critic update -----
    # Current Q(s, a) = w[:state_dim] @ s + w[-1] * a
    q_current = states @ w_old[:-1] + w_old[-1] * actions

    # Next action from the current (pre-update) deterministic policy.
    next_actions = next_states @ theta

    # Q(s', mu(s'))
    q_next = next_states @ w_old[:-1] + w_old[-1] * next_actions

    # Terminal states have no bootstrap term.
    targets = rewards + gamma * (1.0 - dones) * q_next
    td_errors = targets - q_current

    # Mean semi-gradient critic update.
    critic_features = np.concatenate(
        [states, actions[:, None]],
        axis=1
    )
    w_grad = np.mean(td_errors[:, None] * critic_features, axis=0)

    w_new = w_old + alpha_w * w_grad

    # ----- Actor update -----
    # Use the original/pre-update critic weights as required.
    # dQ/da = w_old[-1]
    # d mu(s) / d theta = s
    # => grad_theta J = mean_s [w_old[-1] * s]
    actor_grad = w_old[-1] * np.mean(states, axis=0)

    # Gradient ascent.
    theta_new = theta + alpha_theta * actor_grad

    return (
        np.round(theta_new, 4).tolist(),
        np.round(w_new, 4).tolist()
    )