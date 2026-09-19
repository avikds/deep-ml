import numpy as np

def superposition_reconstruct(W, b, X):
    """
    Compute reconstructed features for the toy superposition model.
    """
    W = np.asarray(W, dtype=float)
    b = np.asarray(b, dtype=float)
    X = np.asarray(X, dtype=float)

    # Encode with W, decode with tied transpose W.T,
    # then add bias and apply ReLU.
    hidden = X @ W.T
    reconstructed = hidden @ W + b
    reconstructed = np.maximum(reconstructed, 0.0)

    return reconstructed.tolist()