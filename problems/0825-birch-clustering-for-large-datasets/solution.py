import numpy as np

def birch_cluster(X, threshold):
    """
    Single-level BIRCH clustering.

    X: array-like of shape (n_samples, n_features)
    threshold: float, max allowed subcluster radius

    Returns:
        list of centroids sorted lexicographically.
    """
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array")

    subclusters = []

    for x in X:
        x = np.asarray(x, dtype=float)

        if not subclusters:
            subclusters.append([
                1,
                x.copy(),
                x * x
            ])
            continue

        # Find closest centroid; first minimum gives smallest index on ties.
        distances = []
        for N, LS, SS in subclusters:
            centroid = LS / N
            distances.append(np.linalg.norm(x - centroid))

        idx = int(np.argmin(distances))

        # Tentatively absorb x.
        N, LS, SS = subclusters[idx]
        new_N = N + 1
        new_LS = LS + x
        new_SS = SS + x * x

        centroid = new_LS / new_N
        variance = new_SS / new_N - centroid * centroid
        variance = np.maximum(variance, 0.0)
        radius = float(np.sqrt(np.sum(variance)))

        if radius <= threshold:
            subclusters[idx] = [new_N, new_LS, new_SS]
        else:
            subclusters.append([
                1,
                x.copy(),
                x * x
            ])

    centroids = [
        (LS / N).tolist()
        for N, LS, SS in subclusters
    ]

    centroids.sort()

    return centroids