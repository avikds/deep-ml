import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


NUMERIC = [
    "age", "fnlwgt", "education-num",
    "capital-gain", "capital-loss", "hours-per-week"
]


def _prepare(X):
    X = X.copy()

    for col in NUMERIC:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    cat_cols = [c for c in X.columns if c not in NUMERIC]

    for col in cat_cols:
        X[col] = X[col].astype(object)
        X[col] = X[col].where(X[col].notna(), "__MISSING__")

    # Reduce the skew of capital gain and loss.
    X["capital-gain"] = np.log1p(
        np.maximum(X["capital-gain"], 0)
    )
    X["capital-loss"] = np.log1p(
        np.maximum(X["capital-loss"], 0)
    )

    return X, cat_cols


def _target(y):
    a = np.asarray(y)

    if a.dtype.kind in "OUS":
        return np.array(
            [int(str(v).strip().startswith(">")) for v in a],
            dtype=np.int64
        )

    return a.astype(np.int64)


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def train(X_train, y_train):
    y = _target(y_train)

    X, cat_cols = _prepare(X_train)

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(
            strategy="median",
            add_indicator=True
        )),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(
            strategy="constant",
            fill_value="__MISSING__"
        )),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipeline, NUMERIC),
            ("cat", categorical_pipeline, cat_cols)
        ],
        sparse_threshold=0
    )

    # Fit feature transformations on training features.
    Xz = preprocessor.fit_transform(X).astype(np.float32)

    indices = np.arange(len(y))
    itr, iva = train_test_split(
        indices,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    xt, xv = Xz[itr], Xz[iva]
    yt, yv = y[itr], y[iva]

    # Model 1: gradient-boosted decision trees.
    hgb = HistGradientBoostingClassifier(
        max_iter=180,
        learning_rate=0.08,
        max_leaf_nodes=31,
        min_samples_leaf=25,
        l2_regularization=2.0,
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=15,
        random_state=42
    )
    hgb.fit(xt, yt)

    # Model 2: randomized trees.
    extra = ExtraTreesClassifier(
        n_estimators=220,
        max_features=0.85,
        min_samples_leaf=3,
        class_weight=None,
        n_jobs=-1,
        random_state=42
    )
    extra.fit(xt, yt)

    # Model 3: regularized linear classifier.
    logistic = LogisticRegression(
        C=0.5,
        max_iter=300,
        solver="lbfgs"
    )
    logistic.fit(xt, yt)

    # Generate validation probabilities for calibration.
    ph = hgb.predict_proba(xv)[:, 1]
    pe = extra.predict_proba(xv)[:, 1]
    pl = logistic.predict_proba(xv)[:, 1]

    # Choose ensemble weights using validation log loss,
    # then calibrate the selected ensemble.
    candidates = [
        (0.65, 0.20, 0.15),
        (0.70, 0.20, 0.10),
        (0.75, 0.15, 0.10),
        (0.55, 0.30, 0.15),
        (0.70, 0.30, 0.00),
        (0.85, 0.15, 0.00)
    ]

    best_loss = float("inf")
    best_weights = candidates[0]
    best_calibrator = None

    for weights in candidates:
        p = (
            weights[0] * ph
            + weights[1] * pe
            + weights[2] * pl
        )
        z = _logit(p).reshape(-1, 1)

        calibrator = LogisticRegression(
            C=10.0,
            max_iter=200,
            solver="lbfgs"
        )
        calibrator.fit(z, yv)

        calibrated = calibrator.predict_proba(z)[:, 1]
        calibrated = np.clip(calibrated, 1e-7, 1 - 1e-7)

        loss = -np.mean(
            yv * np.log(calibrated)
            + (1 - yv) * np.log(1 - calibrated)
        )

        if loss < best_loss:
            best_loss = loss
            best_weights = weights
            best_calibrator = calibrator

    return {
        "preprocessor": preprocessor,
        "hgb": hgb,
        "extra": extra,
        "logistic": logistic,
        "weights": best_weights,
        "calibrator": best_calibrator,
        "_cache_X": None,
        "_cache_p": None
    }


def decide(model, X, costs):
    """
    Bayes-optimal action under the supplied false-positive,
    false-negative, and rejection costs.
    """

    # The same test frame is evaluated under multiple cost regimes.
    # Reuse its calibrated probabilities instead of rerunning models.
    if model["_cache_X"] is X:
        p1 = model["_cache_p"]
    else:
        Xc, _ = _prepare(X)
        Xz = model["preprocessor"].transform(Xc).astype(np.float32)

        ph = model["hgb"].predict_proba(Xz)[:, 1]
        pe = model["extra"].predict_proba(Xz)[:, 1]
        pl = model["logistic"].predict_proba(Xz)[:, 1]

        w = model["weights"]

        combined = (
            w[0] * ph
            + w[1] * pe
            + w[2] * pl
        )

        calibrated = model["calibrator"].predict_proba(
            _logit(combined).reshape(-1, 1)
        )[:, 1]

        p1 = np.clip(calibrated, 1e-8, 1 - 1e-8)

        model["_cache_X"] = X
        model["_cache_p"] = p1

    c_fp = float(costs["c_fp"])
    c_fn = float(costs["c_fn"])
    c_reject = float(costs["c_reject"])

    # Expected cost of each action.
    cost_0 = p1 * c_fn
    cost_1 = (1.0 - p1) * c_fp
    cost_2 = np.full(len(p1), c_reject)

    # Choose the least expensive action.
    risks = np.column_stack((cost_0, cost_1, cost_2))

    return np.argmin(risks, axis=1).astype(np.int64)