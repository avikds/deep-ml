import numpy as np

def impute(X: np.ndarray) -> np.ndarray:
    '''
    Fill in missing values (NaN) in the input array.

    Args:
        X: Array with possible NaN values, shape (n_samples, n_features)

    Returns:
        X_clean: Array with no NaN values, same shape as X
    '''
    X_clean = np.asarray(X, dtype=float).copy()

    # Median of each feature, ignoring NaNs.
    medians = np.nanmedian(X_clean, axis=0)

    # Columns containing only NaNs have a NaN median.
    medians = np.where(np.isfinite(medians), medians, 0.0)

    # Replace NaN and infinite values with the corresponding
    # feature median.
    for j in range(X_clean.shape[1]):
        bad = ~np.isfinite(X_clean[:, j])
        X_clean[bad, j] = medians[j]

    return X_clean