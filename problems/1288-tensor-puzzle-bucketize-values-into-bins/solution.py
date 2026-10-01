import numpy as np

def bucketize(v: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    """Return bucket indices for v given sorted boundaries."""
    v = np.asarray(v)
    boundaries = np.asarray(boundaries)

    # Count how many boundaries each value is greater than or equal to.
    return np.sum(v[:, None] >= boundaries[None, :], axis=1).astype(int)