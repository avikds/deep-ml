import numpy as np

def bagging_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_estimators: int = 10,
    seed: int = 42
) -> np.ndarray:
    """
    Implement a bagging classifier using decision stumps.
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train).astype(int)
    X_test = np.asarray(X_test)

    n_samples, n_features = X_train.shape
    rng = np.random.RandomState(seed)

    test_predictions = []

    for _ in range(n_estimators):
        # Bootstrap sample
        indices = rng.randint(0, n_samples, size=n_samples)
        X_boot = X_train[indices]
        y_boot = y_train[indices]

        best_error = np.inf
        best_feature = 0
        best_threshold = 0.0
        best_direction = 1

        # Train one decision stump
        for feature in range(n_features):
            values = np.unique(X_boot[:, feature])

            # Include thresholds between consecutive values
            if len(values) == 1:
                thresholds = values
            else:
                thresholds = np.concatenate([
                    [values[0]],
                    (values[:-1] + values[1:]) / 2.0,
                    [values[-1]]
                ])

            for threshold in thresholds:
                # Direction 1:
                # x < threshold -> 0, otherwise -> 1
                pred1 = (X_boot[:, feature] >= threshold).astype(int)
                error1 = np.sum(pred1 != y_boot)

                if error1 < best_error:
                    best_error = error1
                    best_feature = feature
                    best_threshold = threshold
                    best_direction = 1

                # Direction -1:
                # x < threshold -> 1, otherwise -> 0
                pred2 = (X_boot[:, feature] < threshold).astype(int)
                error2 = np.sum(pred2 != y_boot)

                if error2 < best_error:
                    best_error = error2
                    best_feature = feature
                    best_threshold = threshold
                    best_direction = -1

        # Predict test set with this stump
        if best_direction == 1:
            pred_test = (
                X_test[:, best_feature] >= best_threshold
            ).astype(int)
        else:
            pred_test = (
                X_test[:, best_feature] < best_threshold
            ).astype(int)

        test_predictions.append(pred_test)

    test_predictions = np.asarray(test_predictions)

    # Majority vote; ties go to class 0.
    votes_for_1 = np.sum(test_predictions, axis=0)
    predictions = (votes_for_1 > n_estimators / 2).astype(int)

    return predictions