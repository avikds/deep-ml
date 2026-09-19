import numpy as np

def train(X_train, y_train, X_val, y_val):
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    X_val = np.asarray(X_val, dtype=np.float64)
    y_val = np.asarray(y_val, dtype=np.float64)

    n, p = X_train.shape

    # Add intercept without regularizing it.
    X1 = np.column_stack([np.ones(n), X_train])

    # Try a broad ridge grid and select alpha using validation R².
    # The validation set is explicitly provided for tuning by the task.
    alphas = np.logspace(-2, 4, 25)

    best_score = -np.inf
    best_w = None

    y_mean = y_train.mean()
    y_centered = y_train - y_mean

    # Scale targets internally for numerical stability.
    y_scale = np.std(y_train)
    if y_scale < 1e-12:
        y_scale = 1.0
    ys = y_centered / y_scale

    for alpha in alphas:
        # Ridge closed form, with intercept excluded from penalty.
        A = X1.T @ X1
        A.flat[1::A.shape[0] + 1] += alpha

        b = X1.T @ ys

        try:
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(A, b, rcond=None)[0]

        pred = y_mean + y_scale * (
            np.column_stack([np.ones(len(X_val)), X_val]) @ w
        )

        ss_res = np.sum((y_val - pred) ** 2)
        ss_tot = np.sum((y_val - y_val.mean()) ** 2)
        score = 1.0 - ss_res / max(ss_tot, 1e-12)

        if score > best_score:
            best_score = score
            best_w = w

    # Refit using the selected regularization strength.
    # Reconstruct alpha from the selected validation score by searching again
    # and retaining the corresponding model.
    best_alpha = None
    best_score = -np.inf

    Xv1 = np.column_stack([np.ones(len(X_val)), X_val])

    for alpha in alphas:
        A = X1.T @ X1
        idx = np.arange(1, p + 1)
        A[idx, idx] += alpha

        b = X1.T @ ys

        try:
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(A, b, rcond=None)[0]

        pred = y_mean + y_scale * (Xv1 @ w)

        ss_res = np.sum((y_val - pred) ** 2)
        ss_tot = np.sum((y_val - y_val.mean()) ** 2)
        score = 1.0 - ss_res / max(ss_tot, 1e-12)

        if score > best_score:
            best_score = score
            best_alpha = alpha

    # Final fit on the training data with the selected ridge strength.
    A = X1.T @ X1
    idx = np.arange(1, p + 1)
    A[idx, idx] += best_alpha
    b = X1.T @ ys

    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(A, b, rcond=None)[0]

    def predict(X):
        X = np.asarray(X, dtype=np.float64)
        Xb = np.column_stack([np.ones(len(X)), X])
        return y_mean + y_scale * (Xb @ w)

    return predict