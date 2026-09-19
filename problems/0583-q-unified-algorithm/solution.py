import numpy as np


def q_sigma_return(
    states: list,
    actions: list,
    rewards: list,
    sigmas: list,
    Q: np.ndarray,
    pi: np.ndarray,
    b: np.ndarray,
    gamma: float
) -> float:
    """
    Compute the n-step Q(sigma) return.
    """

    Q = np.asarray(Q, dtype=float)
    pi = np.asarray(pi, dtype=float)
    b = np.asarray(b, dtype=float)

    n = len(rewards)

    if len(states) != n + 1:
        raise ValueError("states must have length len(rewards) + 1")
    if len(actions) != n + 1:
        raise ValueError("actions must have length len(rewards) + 1")
    if len(sigmas) != n + 1:
        raise ValueError("sigmas must have length len(rewards) + 1")

    if n == 0:
        return float(Q[states[0], actions[0]])

    # Bootstrap from the final state-action pair.
    G = float(Q[states[n], actions[n]])

    # Work backward through the trajectory.
    for k in range(n - 1, -1, -1):
        next_state = states[k + 1]
        next_action = actions[k + 1]

        # Expected Q-value under the target policy.
        expected_q = float(
            np.dot(pi[next_state], Q[next_state])
        )

        # Importance-sampling ratio for the sampled next action.
        behavior_prob = b[next_state, next_action]
        target_prob = pi[next_state, next_action]

        if behavior_prob == 0.0:
            rho = 0.0
        else:
            rho = target_prob / behavior_prob

        # Q(s_{k+1}, a_{k+1})
        q_next = float(Q[next_state, next_action])

        # Unified Q(sigma) coefficient.
        sigma = float(sigmas[k + 1])
        c = sigma * rho + (1.0 - sigma) * target_prob

        # Backward Q(sigma) recursion.
        G = (
            float(rewards[k])
            + gamma * (
                expected_q
                + c * (G - q_next)
            )
        )

    return float(G)