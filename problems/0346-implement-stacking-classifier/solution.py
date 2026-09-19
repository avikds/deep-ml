import numpy as np

def stacking_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    base_classifiers: list,
    meta_classifier,
    n_folds: int = 5
) -> np.ndarray:
    """
    Implement a stacking classifier ensemble using out-of-fold predictions.
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)

    n_samples = len(X_train)
    n_base = len(base_classifiers)

    if n_folds <= 1 or n_folds > n_samples:
        raise ValueError("n_folds must be between 2 and n_samples")

    # Required contiguous folds:
    # folds 0..k-2 have size n_samples // n_folds,
    # final fold receives the remainder.
    fold_size = n_samples // n_folds

    # Out-of-fold meta-features
    oof = np.zeros((n_samples, n_base), dtype=int)

    for fold in range(n_folds):
        start = fold * fold_size

        if fold == n_folds - 1:
            end = n_samples
        else:
            end = start + fold_size

        val_idx = np.arange(start, end)

        train_idx = np.concatenate([
            np.arange(0, start),
            np.arange(end, n_samples)
        ])

        X_tr = X_train[train_idx]
        y_tr = y_train[train_idx]
        X_val = X_train[val_idx]

        for j, clf in enumerate(base_classifiers):
            pred = np.asarray(
                clf(X_tr, y_tr, X_val)
            )

            oof[val_idx, j] = pred.astype(int).reshape(-1)

    # Train meta-classifier on OOF predictions.
    # The classifier function follows the same interface and can therefore
    # return predictions directly when passed the OOF features.
    dummy_X = oof
    meta_train_pred = meta_classifier(
        dummy_X,
        y_train,
        dummy_X
    )

    # Train each base classifier on the complete training set
    # and generate test meta-features.
    test_meta = np.zeros((len(X_test), n_base), dtype=int)

    for j, clf in enumerate(base_classifiers):
        pred = np.asarray(
            clf(X_train, y_train, X_test)
        )
        test_meta[:, j] = pred.astype(int).reshape(-1)

    # Apply the already-fitted meta-model.
    # Since the supplied API makes meta_classifier a function rather than
    # an estimator object, use it once more on the OOF data to obtain the
    # final mapping.
    final_pred = meta_classifier(
        dummy_X,
        y_train,
        test_meta
    )

    return np.asarray(final_pred, dtype=int).reshape(-1)