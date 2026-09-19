import numpy as np

class MyReducer:
    """
    PCA-based dimensionality reduction to 10 dimensions.
    """

    def __init__(self):
        self.n_components = 10
        self.mean_ = None
        self.components_ = None

    def fit(self, X):
        """
        Learn the top principal components from X.
        """
        X = np.asarray(X, dtype=np.float64)

        # Center the training data.
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # Compute the feature covariance matrix.
        # Scaling by n-1 does not affect the eigenvectors.
        covariance = (X_centered.T @ X_centered) / max(1, X.shape[0] - 1)

        # Covariance is symmetric, so eigh is appropriate.
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)

        # Eigenvalues are returned in ascending order.
        # Select eigenvectors corresponding to the largest eigenvalues.
        indices = np.argsort(eigenvalues)[-self.n_components:][::-1]

        self.components_ = eigenvectors[:, indices]

        return self

    def transform(self, X):
        """
        Project X onto the learned principal components.
        """
        X = np.asarray(X, dtype=np.float64)

        X_centered = X - self.mean_

        return X_centered @ self.components_

    def fit_transform(self, X):
        """Fit PCA and transform X."""
        return self.fit(X).transform(X)