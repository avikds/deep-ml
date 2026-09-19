import numpy as np

class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2,
                 max_features='sqrt', random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)

        self.n_features_ = X.shape[1]
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        # Map arbitrary integer labels to 0..n_classes-1 internally.
        self.class_to_index_ = {
            c: i for i, c in enumerate(self.classes_)
        }
        y_encoded = np.array(
            [self.class_to_index_[v] for v in y],
            dtype=np.int64
        )

        self.rng_ = np.random.RandomState(self.random_state)

        # Store the tree as arrays:
        # feature[node], threshold[node], left[node], right[node], value[node]
        self.feature_ = []
        self.threshold_ = []
        self.left_ = []
        self.right_ = []
        self.value_ = []

        self._build_tree(X, y_encoded, np.arange(len(y)))

        return self

    def _num_features_to_try(self):
        if self.max_features is None:
            return self.n_features_

        if isinstance(self.max_features, str):
            if self.max_features == 'sqrt':
                return max(1, int(np.sqrt(self.n_features_)))
            elif self.max_features == 'log2':
                return max(1, int(np.log2(self.n_features_)))
            else:
                raise ValueError(
                    "max_features must be 'sqrt', 'log2', None, or int"
                )

        return min(self.n_features_, int(self.max_features))

    def _gini(self, counts, n):
        if n == 0:
            return 0.0

        p = counts / n
        return 1.0 - np.dot(p, p)

    def _best_split(self, X, y, indices):
        n = len(indices)

        if n < self.min_samples_split:
            return None

        parent_counts = np.bincount(
            y[indices],
            minlength=self.n_classes_
        )

        # Already pure.
        if np.count_nonzero(parent_counts) <= 1:
            return None

        n_try = self._num_features_to_try()

        # Random feature subset at THIS node.
        features = self.rng_.choice(
            self.n_features_,
            size=n_try,
            replace=False
        )

        best_gain = 0.0
        best_feature = None
        best_threshold = None

        parent_gini = self._gini(parent_counts, n)

        for feature in features:
            vals = X[indices, feature]

            # Sort samples by this feature.
            order = np.argsort(vals, kind='quicksort')
            sorted_vals = vals[order]
            sorted_y = y[indices[order]]

            # Left/right class counts.
            left_counts = np.zeros(self.n_classes_, dtype=np.int64)
            right_counts = parent_counts.copy()

            # Only consider thresholds between distinct values.
            for i in range(n - 1):
                cls = sorted_y[i]

                left_counts[cls] += 1
                right_counts[cls] -= 1

                if sorted_vals[i] == sorted_vals[i + 1]:
                    continue

                n_left = i + 1
                n_right = n - n_left

                # Avoid very small child nodes.
                if n_left == 0 or n_right == 0:
                    continue

                g_left = self._gini(left_counts, n_left)
                g_right = self._gini(right_counts, n_right)

                child_gini = (
                    (n_left / n) * g_left +
                    (n_right / n) * g_right
                )

                gain = parent_gini - child_gini

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = (
                        sorted_vals[i] + sorted_vals[i + 1]
                    ) * 0.5

        if best_feature is None:
            return None

        return best_feature, best_threshold

    def _build_tree(self, X, y, indices, depth=0):
        # Majority class at this node.
        counts = np.bincount(
            y[indices],
            minlength=self.n_classes_
        )
        majority = int(np.argmax(counts))

        node_id = len(self.feature_)

        # Initially make this node a leaf.
        self.feature_.append(-1)
        self.threshold_.append(0.0)
        self.left_.append(-1)
        self.right_.append(-1)
        self.value_.append(majority)

        # Stopping conditions.
        if depth >= self.max_depth:
            return node_id

        if len(indices) < self.min_samples_split:
            return node_id

        if np.count_nonzero(counts) <= 1:
            return node_id

        split = self._best_split(X, y, indices)

        if split is None:
            return node_id

        feature, threshold = split

        mask = X[indices, feature] <= threshold

        left_indices = indices[mask]
        right_indices = indices[~mask]

        if len(left_indices) == 0 or len(right_indices) == 0:
            return node_id

        # Build children.
        left_id = self._build_tree(
            X, y, left_indices, depth + 1
        )

        right_id = self._build_tree(
            X, y, right_indices, depth + 1
        )

        self.feature_[node_id] = feature
        self.threshold_[node_id] = threshold
        self.left_[node_id] = left_id
        self.right_[node_id] = right_id

        return node_id

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)

        n_samples = X.shape[0]
        predictions = np.empty(n_samples, dtype=np.int64)

        features = np.asarray(self.feature_, dtype=np.int64)
        thresholds = np.asarray(self.threshold_, dtype=np.float64)
        left = np.asarray(self.left_, dtype=np.int64)
        right = np.asarray(self.right_, dtype=np.int64)
        values = np.asarray(self.value_, dtype=np.int64)

        for i in range(n_samples):
            node = 0

            while features[node] != -1:
                f = features[node]

                if X[i, f] <= thresholds[node]:
                    node = left[node]
                else:
                    node = right[node]

            predictions[i] = self.classes_[values[node]]

        return predictions