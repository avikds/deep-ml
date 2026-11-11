import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import train_test_split


def fit_predict(
    X_train,
    claims_train,
    exposure_train,
    X_test,
    exposure_test
):
    # ------------------------------------------------
    # 1. Prepare data
    # ------------------------------------------------

    cat_cols = [
        "Area",
        "VehBrand",
        "VehGas",
        "Region"
    ]

    num_cols = [
        "VehPower",
        "VehAge",
        "DrivAge",
        "BonusMalus",
        "Density"
    ]

    train = pd.DataFrame(X_train)
    test = pd.DataFrame(X_test)

    # ------------------------------------------------
    # 2. Feature engineering
    # ------------------------------------------------

    for df in (train, test):
        df["LogDensity"] = np.log1p(
            np.maximum(df["Density"].astype(float), 0)
        )

        df["LogVehAge"] = np.log1p(
            np.maximum(df["VehAge"].astype(float), 0)
        )

        df["LogDrivAge"] = np.log1p(
            np.maximum(df["DrivAge"].astype(float), 0)
        )

        df["LogBonusMalus"] = np.log1p(
            np.maximum(df["BonusMalus"].astype(float), 0)
        )

        df["PowerAge"] = (
            df["VehPower"].astype(float) *
            df["VehAge"].astype(float)
        )

        df["AgeBonus"] = (
            df["DrivAge"].astype(float) *
            df["BonusMalus"].astype(float)
        )

    # Consistent categorical encoding.
    for col in cat_cols:
        categories = pd.Index(
            pd.concat(
                [train[col], test[col]],
                ignore_index=True
            ).astype(str).unique()
        )

        train[col] = pd.Categorical(
            train[col].astype(str),
            categories=categories
        )

        test[col] = pd.Categorical(
            test[col].astype(str),
            categories=categories
        )

    # ------------------------------------------------
    # 3. Target and exposure weights
    # ------------------------------------------------

    y = np.asarray(claims_train, dtype=np.float64)
    exp = np.asarray(exposure_train, dtype=np.float64)
    exp_test = np.asarray(exposure_test, dtype=np.float64)

    y = np.maximum(y, 0)
    exp = np.maximum(exp, 1e-8)
    exp_test = np.maximum(exp_test, 1e-8)

    # Annual claim frequency.
    target = y / exp

    # Exposure is the statistical sample weight.
    weights = exp

    # ------------------------------------------------
    # 4. Validation split
    # ------------------------------------------------

    idx_train, idx_valid = train_test_split(
        np.arange(len(y)),
        test_size=0.15,
        random_state=42
    )

    # ------------------------------------------------
    # 5. LightGBM Poisson model
    # ------------------------------------------------

    params = {
        "objective": "poisson",
        "metric": "poisson",

        "n_estimators": 1200,
        "learning_rate": 0.035,

        "num_leaves": 15,
        "max_depth": -1,

        "min_child_samples": 150,

        "subsample": 0.8,
        "subsample_freq": 1,

        "colsample_bytree": 0.9,

        "reg_alpha": 0.5,
        "reg_lambda": 5.0,

        "max_bin": 127,

        "verbosity": -1,
        "n_jobs": -1,

        "random_state": 42
    }

    model = lgb.LGBMRegressor(**params)

    model.fit(
        train.iloc[idx_train],
        target[idx_train],
        sample_weight=weights[idx_train],

        eval_set=[
            (
                train.iloc[idx_valid],
                target[idx_valid]
            )
        ],

        eval_sample_weight=[
            weights[idx_valid]
        ],

        eval_metric="poisson",

        categorical_feature=cat_cols,

        callbacks=[
            lgb.early_stopping(
                stopping_rounds=60,
                verbose=False
            ),
            lgb.log_evaluation(period=0)
        ]
    )

    best_iter = model.best_iteration_

    if best_iter is None or best_iter <= 0:
        best_iter = 500

    # ------------------------------------------------
    # 6. Retrain on all available observations
    # ------------------------------------------------

    final_model = lgb.LGBMRegressor(
        **{
            **params,
            "n_estimators": best_iter
        }
    )

    final_model.fit(
        train,
        target,
        sample_weight=weights,
        categorical_feature=cat_cols
    )

    # ------------------------------------------------
    # 7. Predict annual frequency
    # ------------------------------------------------

    rate = final_model.predict(test)

    rate = np.maximum(rate, 1e-9)
    rate = np.minimum(rate, 5.0)

    # ------------------------------------------------
    # 8. Global frequency calibration
    # ------------------------------------------------

    train_rate = y.sum() / exp.sum()

    predicted_rate = np.sum(rate * exp_test) / exp_test.sum()

    if predicted_rate > 0:
        correction = train_rate / predicted_rate

        # Avoid excessive calibration.
        correction = np.clip(
            correction,
            0.80,
            1.25
        )

        rate *= correction

    # ------------------------------------------------
    # 9. Return expected claim counts
    # ------------------------------------------------

    predictions = rate * exp_test

    return np.maximum(
        predictions,
        1e-9
    )