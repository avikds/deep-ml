import numpy as np


def purged_walk_forward_splits(n_samples: int, n_splits: int, embargo: int) -> list:
    """Build expanding-window train/test splits with an embargo gap."""
    fold_size = n_samples // (n_splits + 1)
    splits = []

    for k in range(1, n_splits + 1):
        test_start = k * fold_size
        test_end = (k + 1) * fold_size if k < n_splits else n_samples

        train_end = max(0, test_start - embargo)

        train_idx = np.arange(0, train_end, dtype=int)
        test_idx = np.arange(test_start, test_end, dtype=int)

        splits.append((train_idx, test_idx))

    return splits