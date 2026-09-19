import numpy as np

def compute_elbo(
    x: list[float],
    q_mean: float,
    q_std: float,
    prior_mean: float,
    prior_std: float,
    likelihood_std: float,
    n_samples: int = 1000
) -> float:
    x = np.asarray(x, dtype=float)

    z = np.random.normal(
        loc=q_mean,
        scale=q_std,
        size=(n_samples, 1)
    )

    # E_q[log p(x | z)]
    log_likelihood = (
        -0.5 * np.log(2.0 * np.pi * likelihood_std**2)
        -0.5 * ((x[None, :] - z) ** 2) / (likelihood_std**2)
    )

    # Sum over observed dimensions
    expected_log_likelihood = np.mean(
        np.sum(log_likelihood, axis=1)
    )

    # E_q[log p(z)]
    log_prior = (
        -0.5 * np.log(2.0 * np.pi * prior_std**2)
        -0.5 * ((z - prior_mean) ** 2) / (prior_std**2)
    )
    expected_log_prior = np.mean(log_prior)

    # H[q]
    entropy = 0.5 * np.log(
        2.0 * np.pi * np.e * q_std**2
    )

    return float(
        expected_log_likelihood
        + expected_log_prior
        + entropy
    )