import numpy as np


def actor_critic_traces(
    num_states: int,
    num_actions: int,
    episode: list,
    gamma: float,
    lambda_w: float,
    lambda_theta: float,
    alpha_w: float,
    alpha_theta: float,
    w_init: np.ndarray,
    theta_init: np.ndarray
) -> tuple:
    """
    Episodic actor-critic with eligibility traces.
    """

    # Work on copies so the caller's arrays are not modified in-place.
    w = np.asarray(w_init, dtype=float).copy()
    theta = np.asarray(theta_init, dtype=float).copy()

    # Eligibility traces.
    z_w = np.zeros(num_states, dtype=float)
    z_theta = np.zeros((num_states, num_actions), dtype=float)

    # Discount accumulator for the actor.
    I = 1.0

    for state, action, reward, next_state, done in episode:
        state = int(state)
        action = int(action)
        reward = float(reward)
        next_state = int(next_state)

        # ---------------------------------------------------------
        # 1. Critic values and TD error
        # ---------------------------------------------------------
        v_current = w[state]

        if done:
            v_next = 0.0
        else:
            v_next = w[next_state]

        delta = reward + gamma * v_next - v_current

        # ---------------------------------------------------------
        # 2. Critic accumulating eligibility trace
        # ---------------------------------------------------------
        z_w *= gamma * lambda_w
        z_w[state] += 1.0

        # ---------------------------------------------------------
        # 3. Actor softmax policy
        # ---------------------------------------------------------
        logits = theta[state] - np.max(theta[state])
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits)

        # Gradient of log pi(a|s) with respect to theta[s, :]
        grad_log_pi = -probs.copy()
        grad_log_pi[action] += 1.0

        # ---------------------------------------------------------
        # 4. Actor accumulating eligibility trace
        # ---------------------------------------------------------
        z_theta *= gamma * lambda_theta
        z_theta[state] += I * grad_log_pi

        # ---------------------------------------------------------
        # 5. Update critic and actor
        # ---------------------------------------------------------
        w += alpha_w * delta * z_w
        theta += alpha_theta * delta * z_theta

        # ---------------------------------------------------------
        # 6. Discount accumulator for next timestep
        # ---------------------------------------------------------
        I *= gamma

        if done:
            break

    return w, theta