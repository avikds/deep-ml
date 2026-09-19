import numpy as np
from sklearn.linear_model import Ridge


def train(X_train, y_train, X_val, y_val):
    """
    Train a regularized regression model and return a prediction callable.
    """

    # Try a broad range of regularization strengths.
    alphas = np.logspace(-2, 4, 40)

    best_alpha = None
    best_r2 = -np.inf

    # Select alpha using the provided validation set.
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)

        val_pred = model.predict(X_val)

        # R^2
        ss_res = np.sum((y_val - val_pred) ** 2)
        ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)

        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        if r2 > best_r2:
            best_r2 = r2
            best_alpha = alpha

    # Train the final model using the best regularization strength.
    final_model = Ridge(alpha=best_alpha)
    final_model.fit(X_train, y_train)

    # Return a callable as required.
    def predict(X):
        return final_model.predict(X)

    return predict