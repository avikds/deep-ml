import numpy as np


def a2c_update(
    num_states: int,
    num_actions: int,
    batch_states: list,
    batch_actions: list,
    batch_rewards: list,
    batch_dones: list,
    bootstrap_states: list,
    gamma: float,
    value_coeff: float,
    entropy_coeff: float,
    alpha_theta: float,
    alpha_w: float,
    theta_init: np.ndarray,
    w_init: np.ndarray
) -> tuple:
    """
    Perform a single A2C batch update from parallel environment data.
    """

    theta = np.asarray(theta_init, dtype=float).copy()
    w = np.asarray(w_init, dtype=float).copy()

    K = len(batch_states)
    total_samples = sum(len(states) for states in batch_states)

    if total_samples == 0:
        return w, theta

    # Accumulated gradients.
    grad_theta = np.zeros_like(theta)
    grad_w = np.zeros_like(w)

    for k in range(K):
        states = batch_states[k]
        actions = batch_actions[k]
        rewards = batch_rewards[k]
        dones = batch_dones[k]

        T = len(states)

        # ---------------------------------------------------------
        # 1. Compute returns by bootstrapping backwards.
        # ---------------------------------------------------------
        G = float(w[bootstrap_states[k]])
        returns = np.zeros(T, dtype=float)

        for t in range(T - 1, -1, -1):
            if dones[t]:
                G = float(rewards[t])
            else:
                G = float(rewards[t]) + gamma * G

            returns[t] = G

        # ---------------------------------------------------------
        # 2. Compute advantages and accumulate gradients.
        # ---------------------------------------------------------
        for t in range(T):
            s = int(states[t])
            a = int(actions[t])

            # Current critic value.
            value = w[s]

            # Advantage.
            advantage = returns[t] - value

            # Critic gradient: d(1/2 * advantage^2) / dw[s]
            # The problem specifies advantage as the error signal
            # and applies value_coeff during the update.
            grad_w[s] += advantage

            # Stable softmax.
            logits = theta[s] - np.max(theta[s])
            exp_logits = np.exp(logits)
            probs = exp_logits / np.sum(exp_logits)

            # Numerical safety for entropy.
            probs_safe = probs + 1e-10
            log_probs = np.log(probs_safe)

            # Score function for the selected action:
            # grad log pi(a|s) = one_hot(a) - pi
            score = -probs.copy()
            score[a] += 1.0

            # Policy gradient weighted by advantage.
            grad_theta[s] += advantage * score

            # -----------------------------------------------------
            # Entropy gradient:
            #
            # dH/dlogit_j = -pi_j * (log(pi_j) + H)
            # -----------------------------------------------------
            entropy = -np.sum(probs * log_probs)

            entropy_grad = -probs * (log_probs + entropy)

            # Add entropy bonus gradient.
            grad_theta[s] += entropy_coeff * entropy_grad

    # -------------------------------------------------------------
    # 3. Average gradients over all samples.
    # -------------------------------------------------------------
    grad_theta /= total_samples
    grad_w /= total_samples

    # -------------------------------------------------------------
    # 4. Single synchronized parameter update.
    # -------------------------------------------------------------
    theta += alpha_theta * grad_theta
    w += alpha_w * value_coeff * grad_w

    return w, theta