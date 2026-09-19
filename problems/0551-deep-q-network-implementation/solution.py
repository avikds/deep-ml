import numpy as np


def dqn_training_step(
    weights: dict,
    target_weights: dict,
    batch: dict,
    gamma: float,
    learning_rate: float
) -> tuple:
    """
    Perform one DQN training step with backpropagation.

    Network:
        input -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> Q-values
    """

    # Make copies so the input weights are not modified in-place.
    W1 = np.asarray(weights["W1"], dtype=float).copy()
    b1 = np.asarray(weights["b1"], dtype=float).copy()
    W2 = np.asarray(weights["W2"], dtype=float).copy()
    b2 = np.asarray(weights["b2"], dtype=float).copy()

    # Batch data.
    states = np.asarray(batch["states"], dtype=float)
    actions = np.asarray(batch["actions"], dtype=int)
    rewards = np.asarray(batch["rewards"], dtype=float)
    next_states = np.asarray(batch["next_states"], dtype=float)
    dones = np.asarray(batch["dones"], dtype=bool)

    batch_size = states.shape[0]

    # =========================================================
    # 1. Forward pass through the online Q-network
    # =========================================================
    z1 = states @ W1 + b1
    h1 = np.maximum(z1, 0.0)
    q_values = h1 @ W2 + b2

    # Q-values for the actions actually taken.
    q_taken = q_values[np.arange(batch_size), actions]

    # =========================================================
    # 2. Target network: compute TD targets
    # =========================================================
    W1_t = np.asarray(target_weights["W1"], dtype=float)
    b1_t = np.asarray(target_weights["b1"], dtype=float)
    W2_t = np.asarray(target_weights["W2"], dtype=float)
    b2_t = np.asarray(target_weights["b2"], dtype=float)

    z1_t = next_states @ W1_t + b1_t
    h1_t = np.maximum(z1_t, 0.0)
    q_next = h1_t @ W2_t + b2_t

    max_q_next = np.max(q_next, axis=1)

    targets = rewards + gamma * max_q_next * (~dones)

    # =========================================================
    # 3. MSE loss
    # =========================================================
    errors = q_taken - targets
    loss = float(np.mean(errors ** 2))

    # =========================================================
    # 4. Backpropagation
    # =========================================================
    #
    # L = mean((q_taken - target)^2)
    # dL/dq_taken = 2 * error / batch_size
    #
    dq_taken = 2.0 * errors / batch_size

    # Gradient of Q output.
    dQ = np.zeros_like(q_values)
    dQ[np.arange(batch_size), actions] = dq_taken

    # Second linear layer:
    # q = h1 @ W2 + b2
    dW2 = h1.T @ dQ
    db2 = np.sum(dQ, axis=0)

    # Backprop through h1.
    dh1 = dQ @ W2.T

    # Backprop through ReLU.
    dz1 = dh1 * (z1 > 0.0)

    # First linear layer:
    # z1 = states @ W1 + b1
    dW1 = states.T @ dz1
    db1 = np.sum(dz1, axis=0)

    # =========================================================
    # 5. Gradient descent update
    # =========================================================
    W1_new = W1 - learning_rate * dW1
    b1_new = b1 - learning_rate * db1
    W2_new = W2 - learning_rate * dW2
    b2_new = b2 - learning_rate * db2

    updated_weights = {
        "W1": W1_new,
        "b1": b1_new,
        "W2": W2_new,
        "b2": b2_new,
    }

    return updated_weights, round(loss, 4)