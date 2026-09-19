import numpy as np
import math

def adaboost_fit(X, y, n_clf):
    n_samples, n_features = X.shape

    w = np.full(n_samples, 1.0 / n_samples)
    clfs = []

    for _ in range(n_clf):
        best_error = float("inf")
        best_clf = None

        for feature_i in range(n_features):
            feature_values = X[:, feature_i]

            for threshold in np.unique(feature_values):
                # polarity = 1
                predictions = np.ones(n_samples)
                predictions[feature_values < threshold] = -1
                error = np.sum(w[predictions != y])

                if error < best_error:
                    best_error = error
                    best_clf = {
                        "polarity": 1,
                        "threshold": threshold,
                        "feature_index": feature_i
                    }

                # polarity = -1
                predictions = np.ones(n_samples)
                predictions[feature_values >= threshold] = -1
                error = np.sum(w[predictions != y])

                if error < best_error:
                    best_error = error
                    best_clf = {
                        "polarity": -1,
                        "threshold": threshold,
                        "feature_index": feature_i
                    }

        # Match the required convention for a perfect stump:
        # alpha = 0.5 * log((1-error) / (error + 1e-10))
        alpha = 0.5 * math.log(
            (1.0 - best_error) / (best_error + 1e-10)
        )

        best_clf["alpha"] = float(alpha)

        # Compute predictions for the selected stump
        feature_values = X[:, best_clf["feature_index"]]
        predictions = np.ones(n_samples)

        if best_clf["polarity"] == 1:
            predictions[feature_values < best_clf["threshold"]] = -1
        else:
            predictions[feature_values >= best_clf["threshold"]] = -1

        # Update weights
        w *= np.exp(-alpha * y * predictions)
        w /= np.sum(w)

        clfs.append(best_clf)

    return clfs