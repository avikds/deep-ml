import numpy as np

def mutual_information(joint_prob: list[list[float]]) -> float:
    """
    Compute the mutual information between two random variables.

    Args:
        joint_prob: 2D joint probability distribution P(X,Y)

    Returns:
        Mutual information I(X;Y)
    """
    pxy = np.asarray(joint_prob, dtype=float)

    # Marginal distributions.
    px = np.sum(pxy, axis=1, keepdims=True)
    py = np.sum(pxy, axis=0, keepdims=True)

    # Only include entries where P(X,Y) > 0 to avoid log(0).
    mask = pxy > 0

    mi = np.sum(
        pxy[mask]
        * np.log(
            pxy[mask]
            / (px @ py)[mask]
        )
    )

    return float(mi)