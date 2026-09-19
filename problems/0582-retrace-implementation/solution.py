import numpy as np


def retrace_targets(
    rewards: np.ndarray,
    Q_values: np.ndarray,
    expected_next_Q: np.ndarray,
    target_pi: np.ndarray,
    behavior_mu: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    lam: float
) -> np.ndarray:
    """
    Compute Retrace(lambda) targets for a trajectory of experience.
    """

    rewards = np.asarray(rewards, dtype=float)
    Q_values = np.asarray(Q_values, dtype=float)
    expected_next_Q = np.asarray(expected_next_Q, dtype=float)
    target_pi = np.asarray(target_pi, dtype=float)
    behavior_mu = np.asarray(behavior_mu, dtype=float)
    dones = np.asarray(dones, dtype=float)

    T = len(rewards)

    if not (
        len(Q_values) == len(expected_next_Q)
        == len(target_pi) == len(behavior_mu)
        == len(dones) == T
    ):
        raise ValueError("All input arrays must have the same length")

    if T == 0:
        return np.array([], dtype=float)

    # Retrace coefficient:
    # c_t = lambda * min(1, pi_t / mu_t)
    ratios = np.zeros(T, dtype=float)

    mask = behavior_mu != 0.0
    ratios[mask] = target_pi[mask] / behavior_mu[mask]

    c = lam * np.minimum(1.0, ratios)

    # TD errors.
    delta = (
        rewards
        + gamma * (1.0 - dones) * expected_next_Q
        - Q_values
    )

    # Backward recursive targets.
    targets = np.zeros(T, dtype=float)

    # Final transition.
    targets[-1] = Q_values[-1] + delta[-1]

    for t in range(T - 2, -1, -1):
        if dones[t] != 0.0:
            # Do not cross an episode boundary.
            targets[t] = Q_values[t] + delta[t]
        else:
            # IMPORTANT: continuation uses c[t + 1].
            targets[t] = (
                Q_values[t]
                + delta[t]
                + gamma
                * c[t + 1]
                * (targets[t + 1] - Q_values[t + 1])
            )

    return np.round(targets, 4)