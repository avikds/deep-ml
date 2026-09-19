import numpy as np


def split_and_baseline(X, y, train_frac, val_frac, test_frac, seed):
    """
    Seeded shuffle split of (X, y), fit a mean baseline on train, evaluate MAE on test.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
    train_frac, val_frac, test_frac : float
        Target fractions. Use int(n * frac) for train and val; remainder -> test.
    seed : int
        RNG seed for the shuffle.

    Returns
    -------
    mae : float
        Mean absolute error of the train-mean baseline on the test set.
    train_idx, val_idx, test_idx : np.ndarray
        1-D integer index arrays (a partition of range(n_samples)).
    """
    n = len(y)

    # Seeded shuffle of row indices.
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    # Required split sizes.
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    # Contiguous blocks: train -> validation -> test.
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    # Mean baseline fitted ONLY on the training targets.
    mu = np.mean(y[train_idx])

    # Constant prediction mu for every test example.
    mae = np.mean(np.abs(y[test_idx] - mu))

    return float(mae), train_idx.astype(int), val_idx.astype(int), test_idx.astype(int)