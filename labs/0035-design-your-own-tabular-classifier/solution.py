import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score


def train(X_train, y_train, X_val, y_val):
    """
    Classical Adult Census binary classifier.

    Returns:
        predict_proba(X): 1-D numpy array containing positive-class scores.
    """

    X_train = X_train.copy()
    X_val = X_val.copy()

    y_train = np.asarray(y_train, dtype=int)
    y_val = np.asarray(y_val, dtype=int)

    # Identify columns by dtype.
    numeric_cols = X_train.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_cols = [
        c for c in X_train.columns
        if c not in numeric_cols
    ]

    # Numeric preprocessing.
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical preprocessing.
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=2
        ))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols)
    ])

    # Logistic regression works particularly well on Adult after
    # one-hot encoding and scaling.
    model = Pipeline([
        ("preprocess", preprocessor),
        ("classifier", LogisticRegression(
            C=1.0,
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
            random_state=0
        ))
    ])

    model.fit(X_train, y_train)

    # Use validation data to choose a monotonic transformation/weighting.
    # For PR-AUC, ranking is the important property.
    val_scores = model.predict_proba(X_val)[:, 1]
    best_ap = average_precision_score(y_val, val_scores)

    # Try a few regularization strengths and retain the best validation
    # ranking. This is inexpensive for the Adult dataset.
    best_model = model

    for C in (0.25, 0.5, 2.0, 4.0):
        candidate = Pipeline([
            ("preprocess", preprocessor),
            ("classifier", LogisticRegression(
                C=C,
                max_iter=2000,
                class_weight="balanced",
                solver="liblinear",
                random_state=0
            ))
        ])

        candidate.fit(X_train, y_train)
        scores = candidate.predict_proba(X_val)[:, 1]
        ap = average_precision_score(y_val, scores)

        if ap > best_ap:
            best_ap = ap
            best_model = candidate

    def predict_proba(X):
        X = X.copy()
        scores = best_model.predict_proba(X)[:, 1]
        return np.asarray(scores, dtype=float)

    return predict_proba