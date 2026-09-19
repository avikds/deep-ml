import numpy as np
import torch
import torch.nn as nn


class FeatureDeconfounder(nn.Module):
    """
    Removes the linear influence of metadata/confounding variables from features.
    """

    def __init__(self):
        super().__init__()
        self.Sigma_inv = None
        self.is_fitted = False

    def fit(self, metadata):
        """
        Precompute:
            Sigma_inv = (X^T X + reg * I)^(-1)
        """
        reg = 1e-5

        if isinstance(metadata, torch.Tensor):
            X = metadata.detach()

            if X.ndim != 2:
                raise ValueError("metadata must be a 2D array/tensor")

            K = X.shape[1]

            # Use float computation for numerical stability.
            X = X.float()

            Sigma = X.T @ X
            I = torch.eye(K, dtype=X.dtype, device=X.device)

            self.Sigma_inv = torch.linalg.inv(Sigma + reg * I)

        else:
            X = np.asarray(metadata)

            if X.ndim != 2:
                raise ValueError("metadata must be a 2D array")

            X = X.astype(np.float64, copy=False)
            K = X.shape[1]

            Sigma = X.T @ X
            I = np.eye(K, dtype=X.dtype)

            self.Sigma_inv = np.linalg.inv(Sigma + reg * I)

        self.is_fitted = True
        return self

    def transform(self, features, metadata):
        """
        Residualize features against metadata:

            beta = Sigma_inv @ X^T @ f
            residual = f - X @ beta

        For PyTorch tensors, gradients flow through `features`.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "FeatureDeconfounder must be fitted before transform()."
            )

        if isinstance(features, torch.Tensor):
            X = metadata.to(
                device=features.device,
                dtype=features.dtype
            )

            Sigma_inv = self.Sigma_inv.to(
                device=features.device,
                dtype=features.dtype
            )

            # beta: (K, D)
            beta = Sigma_inv @ X.T @ features

            # residual: (M, D)
            residual = features - X @ beta

            return residual

        # NumPy path
        F = np.asarray(features)
        X = np.asarray(metadata, dtype=F.dtype)
        Sigma_inv = np.asarray(self.Sigma_inv, dtype=F.dtype)

        beta = Sigma_inv @ X.T @ F
        residual = F - X @ beta

        return residual