import numpy as np

def forward_diffusion(
    x_0: np.ndarray,
    t: int,
    beta_start: float,
    beta_end: float,
    num_timesteps: int,
    noise: np.ndarray
) -> np.ndarray:
    """
    Apply forward diffusion process to add noise to input data.
    """
    betas = np.linspace(beta_start, beta_end, num_timesteps)
    alphas = 1.0 - betas
    alpha_bar = np.cumprod(alphas)

    # t is 1-indexed, so convert to zero-based indexing.
    alpha_bar_t = alpha_bar[t - 1]

    return (
        np.sqrt(alpha_bar_t) * x_0
        + np.sqrt(1.0 - alpha_bar_t) * noise
    )