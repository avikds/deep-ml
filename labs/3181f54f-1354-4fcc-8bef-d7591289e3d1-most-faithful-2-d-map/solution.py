import numpy as np
from sklearn.manifold import TSNE


def embed(X):
    """Return a 2-D embedding optimized to preserve local neighborhoods."""
    X = np.asarray(X, dtype=np.float32)
    n = X.shape[0]

    if n == 0:
        return np.empty((0, 2), dtype=np.float32)

    if n == 1:
        return np.zeros((1, 2), dtype=np.float32)

    if n == 2:
        return np.array([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float32)

    # Centering preserves all Euclidean distances and improves numerics.
    X = X - X.mean(axis=0, keepdims=True)

    # Remove constant features without changing pairwise distances.
    variance = np.var(X, axis=0)
    X = X[:, variance > 1e-12]

    if X.shape[1] == 0:
        # Degenerate input: provide distinct, finite coordinates.
        angle = np.arange(n, dtype=np.float32) * (2.0 * np.pi / n)
        return np.column_stack((np.cos(angle), np.sin(angle)))

    perplexity = min(50.0, max(10.0, (n - 1) / 3.0))

    model = TSNE(
        n_components=2,
        perplexity=perplexity,
        early_exaggeration=12.0,
        learning_rate="auto",
        max_iter=1000,
        n_iter_without_progress=300,
        min_grad_norm=1e-7,
        metric="euclidean",
        init="pca",
        method="barnes_hut",
        angle=0.5,
        random_state=42,
        n_jobs=-1,
    )

    Z = model.fit_transform(X)

    return np.asarray(Z, dtype=np.float32)