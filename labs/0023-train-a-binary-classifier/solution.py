import numpy as np
from sklearn.svm import SVC


def train(X_train, y_train, X_val, y_val):
    """
    Train a binary classifier and return a prediction callable.
    """

    # Small deterministic hyperparameter search.
    candidates = [
        (0.5, "scale"),
        (1.0, "scale"),
        (2.0, "scale"),
        (5.0, "scale"),
        (10.0, "scale"),
        (20.0, "scale"),
        (1.0, 0.01),
        (2.0, 0.01),
        (5.0, 0.01),
        (10.0, 0.01),
    ]

    best_model = None
    best_accuracy = -1.0

    for C, gamma in candidates:
        model = SVC(
            C=C,
            gamma=gamma,
            kernel="rbf",
            random_state=0
        )

        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        accuracy = np.mean(val_pred == y_val)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

    def predict(X):
        return np.asarray(best_model.predict(X), dtype=int)

    return predict