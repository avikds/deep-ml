import numpy as np

def sarsa_lambda_acrobot(
    features: list,
    rewards: list,
    num_weights: int,
    alpha: float,
    gamma: float,
    lam: float,
    initial_weights: np.ndarray,
    trace_type: str = "accumulating"
) -> list:
    """
    Run one episode of Sarsa(lambda) with linear function approximation.
    """
    w = np.asarray(initial_weights, dtype=float).copy()
    e = np.zeros(num_weights, dtype=float)

    if trace_type not in ("accumulating", "replacing"):
        raise ValueError("trace_type must be 'accumulating' or 'replacing'")

    T = len(rewards)

    for t in range(T):
        x = np.asarray(features[t], dtype=float)

        # Q(S_t, A_t)
        q = np.dot(w, x)

        # Terminal after the final transition.
        if t == T - 1:
            q_next = 0.0
        else:
            x_next = np.asarray(features[t + 1], dtype=float)
            q_next = np.dot(w, x_next)

        # TD error.
        delta = rewards[t] + gamma * q_next - q

        # Eligibility trace update.
        e *= gamma * lam

        if trace_type == "accumulating":
            e += x
        else:
            # Replacing trace: active features are set to 1.
            e[x != 0] = 1.0

        # Semi-gradient Sarsa(lambda) weight update.
        w += alpha * delta * e

    return w.tolist()