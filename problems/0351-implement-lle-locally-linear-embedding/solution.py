import numpy as np

def lle(X: np.ndarray, n_neighbors: int, n_components: int) -> np.ndarray:
    """
    Perform Locally Linear Embedding for dimensionality reduction.
    """
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array")

    n_samples, n_features = X.shape

    if n_neighbors <= 0 or n_neighbors >= n_samples:
        raise ValueError("n_neighbors must be in [1, n_samples - 1]")

    if n_components <= 0 or n_components >= n_samples:
        raise ValueError("n_components must be in [1, n_samples - 1]")

    # Pairwise squared Euclidean distances
    sq_norms = np.sum(X * X, axis=1)
    distances = (
        sq_norms[:, None]
        + sq_norms[None, :]
        - 2.0 * (X @ X.T)
    )
    distances = np.maximum(distances, 0.0)

    # Reconstruction weight matrix
    W = np.zeros((n_samples, n_samples), dtype=float)

    for i in range(n_samples):
        # Exclude the point itself
        dist_i = distances[i].copy()
        dist_i[i] = np.inf

        neighbors = np.argsort(dist_i)[:n_neighbors]

        # Local difference matrix: neighbors - x_i
        Z = X[neighbors] - X[i]

        # Local covariance / Gram matrix
        C = Z @ Z.T

        # Regularization for numerical stability
        C = C + 1e-3 * np.eye(n_neighbors)

        ones = np.ones(n_neighbors)

        try:
            w = np.linalg.solve(C, ones)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(C, ones, rcond=None)[0]

        # Normalize so weights sum to 1
        w_sum = np.sum(w)

        if abs(w_sum) < 1e-12:
            w = ones / n_neighbors
        else:
            w = w / w_sum

        W[i, neighbors] = w

    # Compute M = (I-W)^T (I-W)
    I = np.eye(n_samples)
    M = (I - W).T @ (I - W)

    # Symmetrize for numerical stability
    M = 0.5 * (M + M.T)

    # Smallest eigenvectors give the LLE embedding.
    eigenvalues, eigenvectors = np.linalg.eigh(M)

    order = np.argsort(eigenvalues)
    eigenvectors = eigenvectors[:, order]

    # First eigenvector corresponds to the zero eigenvalue / constant vector.
    Y = eigenvectors[:, 1:n_components + 1]

    # Deterministic sign convention:
    # make the sum of each component positive.
    for j in range(Y.shape[1]):
        if np.sum(Y[:, j]) < 0:
            Y[:, j] *= -1.0

    return Y