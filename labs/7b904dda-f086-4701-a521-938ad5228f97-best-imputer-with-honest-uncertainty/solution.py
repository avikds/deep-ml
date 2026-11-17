import time
import warnings
import numpy as np

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor

try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None


def impute(X_missing):
    """
    Conditional imputation with MAR correction and predictive uncertainty.

    Returns:
        mean: predictions at missing entries; observed entries unchanged.
        var: predictive variance at missing entries.
    """
    start = time.perf_counter()
    deadline = start + 54.0

    X = np.asarray(X_missing, dtype=np.float64)
    n, d = X.shape
    hidden = np.isnan(X)

    mean = X.copy()
    var = np.ones((n, d), dtype=np.float64)

    if not np.any(hidden):
        return mean, var

    # Initial fallback values.
    col_means = np.nanmean(X, axis=0)
    col_means = np.nan_to_num(col_means)
    col_vars = np.nanvar(X, axis=0)
    col_vars = np.maximum(col_vars, 1e-4)

    for j in range(d):
        mean[hidden[:, j], j] = col_means[j]
        var[hidden[:, j], j] = col_vars[j]

    # Fully observed columns identify possible MAR drivers.
    complete_cols = np.flatnonzero(~np.any(hidden, axis=0))

    # Estimate observation propensities for each target column.
    propensity_weights = np.ones((n, d), dtype=np.float64)

    if len(complete_cols) >= 1:
        C = X[:, complete_cols]

        for j in range(d):
            if not np.any(hidden[:, j]):
                continue

            observed_j = (~hidden[:, j]).astype(int)

            if np.unique(observed_j).size < 2:
                continue

            try:
                prop = LogisticRegression(
                    C=1.0,
                    max_iter=200,
                    solver="lbfgs"
                )
                prop.fit(C, observed_j)

                probability = prop.predict_proba(C)[:, 1]
                probability = np.clip(probability, 0.15, 0.99)

                weights = 1.0 / probability
                weights = np.clip(weights, 0.5, 4.0)
                weights /= np.mean(weights[observed_j == 1])

                propensity_weights[:, j] = weights

            except Exception:
                pass

    # Create predictor matrices with missingness indicators.
    def make_features(columns):
        A = X[:, columns]
        mask = np.isnan(A)

        if A.shape[1] == 0:
            return np.zeros((n, 1), dtype=np.float64)

        medians = np.zeros(A.shape[1], dtype=np.float64)

        for h in range(A.shape[1]):
            values = A[~mask[:, h], h]
            if len(values):
                medians[h] = np.median(values)

        filled = np.where(mask, medians[None, :], A)
        return np.concatenate(
            [filled, mask.astype(np.float64)],
            axis=1
        )

    def make_model(name, seed, train_size):
        if name == "ridge":
            return Ridge(alpha=5.0)

        if name == "hist":
            return HistGradientBoostingRegressor(
                max_iter=110,
                learning_rate=0.055,
                max_leaf_nodes=15,
                min_samples_leaf=max(20, min(45, train_size // 25)),
                l2_regularization=5.0,
                early_stopping=True,
                validation_fraction=0.12,
                n_iter_no_change=12,
                random_state=seed
            )

        if name == "extra":
            return ExtraTreesRegressor(
                n_estimators=80,
                min_samples_leaf=5,
                max_features=0.85,
                n_jobs=1,
                random_state=seed
            )

        if name == "light":
            return LGBMRegressor(
                n_estimators=150,
                learning_rate=0.04,
                num_leaves=15,
                max_depth=-1,
                min_child_samples=30,
                reg_lambda=5.0,
                verbosity=-1,
                n_jobs=1,
                random_state=seed
            )

        return None

    model_names = ["ridge", "hist", "extra"]

    if LGBMRegressor is not None:
        model_names.append("light")

    rng = np.random.RandomState(42)

    for j in range(d):
        if not np.any(hidden[:, j]):
            continue

        if time.perf_counter() > deadline - 2.0:
            break

        columns = np.array([c for c in range(d) if c != j])
        F = make_features(columns)

        observed = np.flatnonzero(~hidden[:, j])
        missing = np.flatnonzero(hidden[:, j])

        if len(observed) < 50:
            continue

        y = X[observed, j]
        w = propensity_weights[observed, j]

        # Split observed records into training and validation sets.
        perm = rng.permutation(len(observed))
        n_val = min(
            max(50, int(0.22 * len(observed))),
            len(observed) - 40
        )

        val_local = perm[:n_val]
        train_local = perm[n_val:]

        train_idx = observed[train_local]
        val_idx = observed[val_local]

        X_train = F[train_idx]
        X_val = F[val_idx]

        y_train = X[train_idx, j]
        y_val = X[val_idx, j]

        w_train = propensity_weights[train_idx, j]
        w_val = propensity_weights[val_idx, j]

        w_train = w_train / max(np.mean(w_train), 1e-12)
        w_val = w_val / max(np.mean(w_val), 1e-12)

        # Fit and compare candidate predictors.
        records = []

        for name in model_names:
            if time.perf_counter() > deadline - 12.0:
                break

            try:
                model = make_model(name, 42 + j, len(train_idx))

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(
                        X_train,
                        y_train,
                        sample_weight=w_train
                    )

                pred = model.predict(X_val)
                mse = float(
                    np.average((y_val - pred) ** 2, weights=w_val)
                )

                if np.isfinite(mse):
                    records.append((mse, name, model, pred))

            except Exception:
                continue

        if not records:
            continue

        records.sort(key=lambda z: z[0])

        # Blend the best two candidates when validation supports it.
        first = records[0]
        second = records[1] if len(records) > 1 else None

        if second is not None and second[0] < first[0] * 1.20:
            best_alpha = 1.0
            best_mse = first[0]

            for alpha in np.linspace(0.0, 1.0, 11):
                blended = (
                    alpha * first[3]
                    + (1.0 - alpha) * second[3]
                )
                mse = float(
                    np.average((y_val - blended) ** 2, weights=w_val)
                )

                if mse < best_mse:
                    best_mse = mse
                    best_alpha = alpha

            selected = [(first, best_alpha)]

            if best_alpha < 1.0:
                selected.append((second, 1.0 - best_alpha))

        else:
            selected = [(first, 1.0)]

        val_prediction = sum(
            weight * record[3]
            for record, weight in selected
        )

        residual = y_val - val_prediction
        bias = float(np.average(residual, weights=w_val))
        residual = residual - bias

        # Estimate the global conditional prediction error.
        base_var = float(
            np.average(residual ** 2, weights=w_val)
        )
        base_var = max(0.90 * base_var, 1e-4)

        # Refit selected predictors using all observed target values.
        final_predictions = []
        final_weights = []

        for record, blend_weight in selected:
            if blend_weight <= 0:
                continue

            if time.perf_counter() > deadline - 2.0:
                # Retain the validation-trained model if time is short.
                final_model = record[2]
            else:
                try:
                    final_model = make_model(
                        record[1], 123 + j, len(observed)
                    )

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        final_model.fit(
                            F[observed],
                            X[observed, j],
                            sample_weight=propensity_weights[observed, j]
                        )

                except Exception:
                    final_model = record[2]

            pred_hidden = final_model.predict(F[missing])

            final_predictions.append(pred_hidden)
            final_weights.append(blend_weight)

        if final_predictions:
            total_weight = sum(final_weights)
            prediction = sum(
                a * b for a, b in zip(final_weights, final_predictions)
            ) / max(total_weight, 1e-12)

            mean[missing, j] = prediction + bias

        # Estimate local variance from validation residuals.
        local_variance = np.full(
            len(missing), base_var, dtype=np.float64
        )

        if (
            len(val_idx) >= 180
            and time.perf_counter() < deadline - 3.0
        ):
            try:
                variance_target = np.log(
                    np.maximum(residual ** 2, base_var * 0.02)
                )

                variance_model = HistGradientBoostingRegressor(
                    max_iter=55,
                    learning_rate=0.05,
                    max_leaf_nodes=7,
                    min_samples_leaf=max(35, len(val_idx) // 15),
                    l2_regularization=10.0,
                    early_stopping=True,
                    random_state=99 + j
                )

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    variance_model.fit(
                        X_val,
                        variance_target,
                        sample_weight=w_val
                    )

                raw_val = np.exp(
                    np.clip(variance_model.predict(X_val), -9.0, 5.0)
                )
                raw_missing = np.exp(
                    np.clip(
                        variance_model.predict(F[missing]),
                        -9.0, 5.0
                    )
                )

                calibration = base_var / max(
                    float(np.average(raw_val, weights=w_val)),
                    1e-8
                )

                raw_missing *= calibration

                # Shrink the local estimates toward the global estimate.
                local_variance = (
                    0.65 * base_var
                    + 0.35 * raw_missing
                )

                local_variance = np.clip(
                    local_variance,
                    0.20 * base_var,
                    5.0 * base_var
                )

            except Exception:
                pass

        var[missing, j] = np.maximum(local_variance, 1e-4)

    # Preserve every observed input value exactly.
    mean[~hidden] = X[~hidden]

    # Final safety checks.
    mean[hidden] = np.nan_to_num(
        mean[hidden], nan=0.0, posinf=0.0, neginf=0.0
    )
    var[hidden] = np.nan_to_num(
        var[hidden], nan=1.0, posinf=1.0, neginf=1.0
    )
    var[hidden] = np.maximum(var[hidden], 1e-4)

    return mean, var