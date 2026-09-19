import numpy as np

def rainbow_dqn_loss(
    online_V_logits: np.ndarray,
    online_A_logits: np.ndarray,
    target_V_logits: np.ndarray,
    target_A_logits: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    gamma_n: float,
    v_min: float,
    v_max: float,
    n_atoms: int
) -> tuple:
    """
    Compute the Rainbow DQN training loss combining dueling architecture,
    double Q-learning, and distributional (categorical) Bellman projection.

    Returns:
        Tuple of (losses, priorities, next_actions)
    """

    online_V_logits = np.asarray(online_V_logits, dtype=float)
    online_A_logits = np.asarray(online_A_logits, dtype=float)
    target_V_logits = np.asarray(target_V_logits, dtype=float)
    target_A_logits = np.asarray(target_A_logits, dtype=float)

    actions = np.asarray(actions, dtype=int)
    rewards = np.asarray(rewards, dtype=float)
    dones = np.asarray(dones, dtype=float)

    batch_size = online_V_logits.shape[0]
    eps = 1e-8

    # ----- 1. Dueling combination -----
    # Q_logits(s,a,z) = V_logits(s,z)
    #                  + A_logits(s,a,z) - mean_a A_logits(s,a,z)
    online_Q_logits = (
        online_V_logits[:, None, :]
        + online_A_logits
        - np.mean(online_A_logits, axis=1, keepdims=True)
    )

    target_Q_logits = (
        target_V_logits[:, None, :]
        + target_A_logits
        - np.mean(target_A_logits, axis=1, keepdims=True)
    )

    # ----- 2. Softmax over atoms -----
    def softmax(x, axis=-1):
        x = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    online_probs = softmax(online_Q_logits, axis=-1)
    target_probs = softmax(target_Q_logits, axis=-1)

    # Fixed categorical support.
    support = np.linspace(v_min, v_max, n_atoms)
    delta_z = (v_max - v_min) / (n_atoms - 1)

    # ----- 3. Double action selection -----
    # Expected Q-values under the online distribution.
    online_q_values = np.sum(
        online_probs * support[None, None, :],
        axis=-1
    )

    next_actions = np.argmax(online_q_values, axis=1)

    # ----- 4. Target distribution for selected next action -----
    batch_idx = np.arange(batch_size)
    next_target_probs = target_probs[batch_idx, next_actions]

    # ----- 5. Distributional Bellman projection -----
    projected = np.zeros((batch_size, n_atoms), dtype=float)

    # Bellman-transformed atom locations.
    tz = rewards[:, None] + gamma_n * (1.0 - dones[:, None]) * support[None, :]
    tz = np.clip(tz, v_min, v_max)

    b = (tz - v_min) / delta_z
    lower = np.floor(b).astype(int)
    upper = np.ceil(b).astype(int)

    # Distribute probability between neighboring atoms.
    for i in range(batch_size):
        for j in range(n_atoms):
            p = next_target_probs[i, j]

            if lower[i, j] == upper[i, j]:
                projected[i, lower[i, j]] += p
            else:
                projected[i, lower[i, j]] += p * (upper[i, j] - b[i, j])
                projected[i, upper[i, j]] += p * (b[i, j] - lower[i, j])

    # ----- 6. Cross-entropy for taken actions -----
    chosen_online_probs = online_probs[batch_idx, actions]
    losses = -np.sum(
        projected * np.log(np.clip(chosen_online_probs, eps, 1.0)),
        axis=1
    )

    # ----- 7. Priorities -----
    priorities = losses + 1e-6

    return (
        np.round(losses, 4).tolist(),
        np.round(priorities, 4).tolist(),
        next_actions.astype(int).tolist()
    )