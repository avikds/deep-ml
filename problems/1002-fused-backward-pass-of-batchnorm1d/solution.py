import numpy as np

def batchnorm1d_backward_fused(
    x: np.ndarray,
    bngain: np.ndarray,
    dhpreact: np.ndarray,
    eps: float = 1e-5
) -> np.ndarray:
    """
    Compute the fused backward pass through a BatchNorm1d layer.
    """
    x = np.asarray(x, dtype=float)
    bngain = np.asarray(bngain, dtype=float)
    dhpreact = np.asarray(dhpreact, dtype=float)

    N = x.shape[0]

    # Batch statistics with Bessel's correction.
    mu = np.mean(x, axis=0, keepdims=True)
    var = np.var(x, axis=0, ddof=1, keepdims=True)

    # Inverse standard deviation.
    sigma_inv = 1.0 / np.sqrt(var + eps)

    # Normalized input.
    xhat = (x - mu) * sigma_inv

    # Fused BatchNorm backward expression.
    dhprebn = (
        bngain
        * sigma_inv
        / N
        * (
            N * dhpreact
            - np.sum(dhpreact, axis=0, keepdims=True)
            - (N / (N - 1))
            * xhat
            * np.sum(dhpreact * xhat, axis=0, keepdims=True)
        )
    )

    return dhprebn