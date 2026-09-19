import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score


def train_ensemble(base_models, X_val, y_val):
    """
    Stack the pretrained classifiers using their probability outputs and
    hard predictions. Base models are never retrained.
    """
    X_val = np.asarray(X_val)
    y_val = np.asarray(y_val).astype(int)

    def make_features(models, X):
        prob_parts = []
        pred_parts = []

        for model in models:
            p = np.asarray(model.predict_proba(X), dtype=float)
            pred = np.asarray(model.predict(X), dtype=int)

            # Normalize defensively in case a classifier returns tiny
            # floating-point deviations from a probability sum of 1.
            p_sum = p.sum(axis=1, keepdims=True)
            p = p / np.maximum(p_sum, 1e-12)

            prob_parts.append(p)

            # One-hot encode the hard prediction. This explicitly exposes
            # model disagreement to the meta learner.
            one_hot = np.zeros((len(X), 10), dtype=float)
            one_hot[np.arange(len(X)), pred] = 1.0
            pred_parts.append(one_hot)

        P = np.concatenate(prob_parts, axis=1)   # K * 10
        H = np.concatenate(pred_parts, axis=1)   # K * 10

        # Pairwise agreement indicators.
        hard_preds = np.column_stack(
            [np.asarray(m.predict(X), dtype=int) for m in models]
        )
        agreements = []
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                agreements.append(
                    (hard_preds[:, i] == hard_preds[:, j]).astype(float)[:, None]
                )

        A = np.concatenate(agreements, axis=1) if agreements else np.empty(
            (len(X), 0), dtype=float
        )

        # Confidence / entropy information is useful for distinguishing
        # confident disagreements from uncertain disagreements.
        extras = []
        for p in prob_parts:
            extras.append(p.max(axis=1, keepdims=True))
            entropy = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum(axis=1)
            extras.append(entropy[:, None])

        E = np.concatenate(extras, axis=1)

        return np.concatenate([P, H, A, E], axis=1)

    X_meta = make_features(base_models, X_val)

    # Choose regularization using only the supplied validation set via
    # stratified CV, then refit on all validation examples.
    candidates = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    best_C = candidates[0]
    best_score = -np.inf

    for C in candidates:
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=C,
                max_iter=3000,
                multi_class="multinomial",
                solver="lbfgs",
                random_state=42,
            ),
        )

        score = cross_val_score(
            clf,
            X_meta,
            y_val,
            cv=cv,
            scoring="accuracy",
            n_jobs=1,
        ).mean()

        if score > best_score:
            best_score = score
            best_C = C

    meta = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=best_C,
            max_iter=3000,
            multi_class="multinomial",
            solver="lbfgs",
            random_state=42,
        ),
    )

    meta.fit(X_meta, y_val)

    def predict(X):
        X = np.asarray(X)
        features = make_features(base_models, X)
        return np.asarray(meta.predict(features), dtype=int)

    return predict