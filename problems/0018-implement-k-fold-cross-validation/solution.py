import numpy as np
from typing import List, Tuple

def k_fold_cross_validation(
    n_samples: int,
    k: int = 5,
    shuffle: bool = True
) -> List[Tuple[List[int], List[int]]]:
    """
    Generate train/test index splits for k-fold cross-validation.
    """

    if k <= 1 or k > n_samples:
        raise ValueError("k must be greater than 1 and at most n_samples")

    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    # First folds receive one extra sample when n_samples
    # is not evenly divisible by k.
    base_size = n_samples // k
    remainder = n_samples % k

    fold_sizes = [
        base_size + (1 if i < remainder else 0)
        for i in range(k)
    ]

    folds = []
    start = 0

    for size in fold_sizes:
        folds.append(indices[start:start + size].tolist())
        start += size

    splits = []

    for i in range(k):
        test_indices = folds[i]

        train_indices = []
        for j in range(k):
            if j != i:
                train_indices.extend(folds[j])

        splits.append((train_indices, test_indices))

    return splits