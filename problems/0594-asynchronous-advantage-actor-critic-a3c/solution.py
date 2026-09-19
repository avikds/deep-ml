import numpy as np

def a3c_update(
    worker_trajectories: list,
    policy_params: list,
    value_params: list,
    n_steps: int,
    gamma: float,
    alpha_policy: float,
    alpha_value: float,
    entropy_coeff: float
) -> dict:
    """
    Simulate the A3C update across multiple workers.
    """

    policy = np.asarray(policy_params, dtype=float).copy()
    values = np.asarray(value_params, dtype=float).copy()

    def softmax(logits):
        z = logits - np.max(logits)
        e = np.exp(z)
        return e / np.sum(e)

    for trajectory in worker_trajectories:
        T = len(trajectory)

        # Snapshot of the shared parameters for this worker.
        policy_old = policy.copy()
        values_old = values.copy()

        # Compute n-step returns using the worker's parameter snapshot.
        returns = np.zeros(T, dtype=float)

        for t in range(T):
            G = 0.0
            discount = 1.0
            end = min(t + n_steps, T)

            for k in range(t, end):
                G += discount * trajectory[k][2]
                discount *= gamma

            if end < T:
                bootstrap_state = trajectory[end][0]
                G += discount * values_old[bootstrap_state]

            returns[t] = G

        # Accumulate gradients for the whole worker trajectory.
        policy_grad = np.zeros_like(policy)
        value_grad = np.zeros_like(values)

        for t, (state, action, _) in enumerate(trajectory):
            probs = softmax(policy_old[state])

            # Advantage computed from the pre-update value parameters.
            advantage = returns[t] - values_old[state]

            # Gradient of log pi(a|s).
            grad_log_pi = -probs
            grad_log_pi[action] += 1.0

            policy_grad[state] += advantage * grad_log_pi

            # Entropy regularization.
            if entropy_coeff != 0.0:
                log_probs = np.log(np.clip(probs, 1e-12, 1.0))
                entropy_grad = probs * (
                    np.sum(probs * log_probs) - log_probs
                )
                policy_grad[state] += entropy_coeff * entropy_grad

            # Value gradient: sum of (return - V).
            value_grad[state] += advantage

        # Apply the worker's accumulated update asynchronously.
        policy += alpha_policy * policy_grad
        values += alpha_value * value_grad

    return {
        "policy_params": np.round(policy, 4).tolist(),
        "value_params": np.round(values, 4).tolist()
    }