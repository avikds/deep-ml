import numpy as np

def grpo_objective(
    rhos,
    A,
    pi_theta_old,
    pi_theta_ref,
    epsilon=0.2,
    beta=0.01
) -> float:
    rhos = np.asarray(rhos, dtype=float)
    A = np.asarray(A, dtype=float)
    pi_theta_old = np.asarray(pi_theta_old, dtype=float)
    pi_theta_ref = np.asarray(pi_theta_ref, dtype=float)

    # PPO-style clipped surrogate
    clipped_rhos = np.clip(rhos, 1.0 - epsilon, 1.0 + epsilon)

    surrogate = np.minimum(
        rhos * A,
        clipped_rhos * A
    )

    # Current policy probability: pi_theta = rho * pi_theta_old
    pi_theta = rhos * pi_theta_old

    # Importance-weighted KL estimator:
    # rho * (pi_ref/pi_theta - log(pi_ref/pi_theta) - 1)
    ratio_ref = pi_theta_ref / np.maximum(pi_theta, 1e-15)

    kl = rhos * (
        ratio_ref
        - np.log(np.maximum(ratio_ref, 1e-15))
        - 1.0
    )

    objective = np.mean(surrogate) - beta * np.mean(kl)

    return float(objective)