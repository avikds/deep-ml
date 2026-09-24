import numpy as np
from typing import Tuple

def find_best_split(X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
    """Return the (feature_index, threshold) that minimises weighted Gini impurity."""
    n_samples, n_features = X.shape

    best_feature = -1
    best_threshold = 0.0
    best_gini = float("inf")

    for feature in range(n_features):
        # Scan thresholds in ascending order; skip the maximum so both
        # child nodes are non-empty.
        thresholds = np.sort(np.unique(X[:, feature]))

        for threshold in thresholds[:-1]:
            left = X[:, feature] <= threshold
            right = ~left

            n_left = np.sum(left)
            n_right = np.sum(right)

            # Gini impurity for the left child.
            p_left = np.mean(y[left])
            gini_left = 2.0 * p_left * (1.0 - p_left)

            # Gini impurity for the right child.
            p_right = np.mean(y[right])
            gini_right = 2.0 * p_right * (1.0 - p_right)

            # Weighted Gini impurity.
            weighted_gini = (
                (n_left / n_samples) * gini_left
                + (n_right / n_samples) * gini_right
            )

            # Strict comparison preserves the first split encountered on ties.
            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_feature = feature
                best_threshold = float(threshold)

    return int(best_feature), best_threshold