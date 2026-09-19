import numpy as np
from sklearn.decomposition import PCA


class MyReducer:
    """
    Implement dimensionality reduction to 10 dimensions.
    """

    def __init__(self):
        self.n_components = 10
        self.reducer = PCA(
            n_components=self.n_components,
            svd_solver="full"
        )

    def fit(self, X):
        """
        Learn the reduction from training data.
        """
        self.reducer.fit(X)
        return self

    def transform(self, X):
        """
        Apply the learned reduction to data.
        """
        return self.reducer.transform(X)

    def fit_transform(self, X):
        """Fit and transform in one step."""
        return self.fit(X).transform(X)