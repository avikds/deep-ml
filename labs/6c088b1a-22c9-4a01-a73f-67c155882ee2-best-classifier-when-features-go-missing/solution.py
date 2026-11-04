import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier

def train(X_train, y_train):
    rng = np.random.default_rng(42)

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.int64)
    n = len(X)

    observed = np.isfinite(X)

    # Estimate pixel statistics using observed training pixels only.
    counts = observed.sum(axis=0)
    sums = np.where(observed, X, 0.0).sum(axis=0)

    mean = sums / np.maximum(counts, 1)
    global_mean = float(np.sum(sums) / max(1, observed.sum()))
    mean = np.where(counts > 0, mean, global_mean)

    centered = np.where(observed, X - mean, 0.0)
    variance = (centered ** 2).sum(axis=0) / np.maximum(counts, 1)
    std = np.sqrt(variance)
    std = np.maximum(std, 1.0)

    def make_features(A):
        mask = np.isfinite(A)
        filled = np.where(mask, A, mean)
        pixels = (filled - mean) / std
        # Include a weak missingness indicator.
        return np.concatenate(
            [pixels, mask.astype(np.float64) * 0.35],
            axis=1
        )

    # Train on the original data plus artificially masked variants.
    feature_sets = [make_features(X)]
    label_sets = [y]

    original_fraction = observed.mean(axis=1)

    # Include a broad range of missingness levels.
    for missing_rate in (0.15, 0.35, 0.55, 0.75, 0.90):
        target_fraction = 1.0 - missing_rate

        keep_probability = np.minimum(
            1.0,
            target_fraction / np.maximum(original_fraction, 1e-6)
        )

        keep = rng.random((n, 64)) < keep_probability[:, None]
        new_mask = observed & keep
        masked = np.where(new_mask, X, np.nan)

        feature_sets.append(make_features(masked))
        label_sets.append(y)

    # Add examples with a hidden contiguous 4x4 region.
    for _ in range(2):
        block_mask = observed.copy()

        for i in range(n):
            r = int(rng.integers(0, 5))
            c = int(rng.integers(0, 5))
            indices = np.arange(r, r + 4)[:, None] * 8
            indices = indices + np.arange(c, c + 4)[None, :]
            block_mask[i, indices.ravel()] = False

        # Sometimes combine the block with scattered missing pixels.
        keep = rng.random((n, 64)) < 0.8
        block_mask &= keep

        masked = np.where(block_mask, X, np.nan)
        feature_sets.append(make_features(masked))
        label_sets.append(y)

    X_aug = np.concatenate(feature_sets, axis=0)
    y_aug = np.concatenate(label_sets, axis=0)

    # RBF SVC is effective on small handwritten digit datasets.
    svm = SVC(
        C=8.0,
        kernel="rbf",
        gamma="scale",
        probability=True,
        cache_size=512,
        random_state=42
    )
    svm.fit(X_aug, y_aug)

    # Randomized trees provide a complementary probability estimate.
    trees = ExtraTreesClassifier(
        n_estimators=300,
        max_features=0.8,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )
    trees.fit(X_aug, y_aug)

    prior = np.bincount(y, minlength=10).astype(np.float64)
    prior /= prior.sum()

    def predict_proba(X_test):
        A = np.asarray(X_test, dtype=np.float64)
        mask = np.isfinite(A)
        observed_count = mask.sum(axis=1)

        features = make_features(A)

        p_svm = svm.predict_proba(features)
        p_tree = trees.predict_proba(features)

        # Align tree class probabilities to labels 0..9.
        tree_probs = np.zeros((len(A), 10), dtype=np.float64)
        tree_probs[:, trees.classes_.astype(int)] = p_tree

        # Ensemble the two calibrated probability estimates.
        P = 0.85 * p_svm + 0.15 * tree_probs

        # Shrink predictions toward the empirical prior when very
        # little image information is available.
        reliability = np.clip(
            observed_count / 20.0,
            0.0,
            1.0
        )
        reliability = 0.35 + 0.65 * reliability

        P = (
            reliability[:, None] * P
            + (1.0 - reliability[:, None]) * prior[None, :]
        )

        # An entirely missing image contains no class information.
        empty = observed_count == 0
        P[empty] = prior

        P = np.maximum(P, 1e-12)
        P /= P.sum(axis=1, keepdims=True)

        return P

    return predict_proba