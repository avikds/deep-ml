import numpy as np

from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def train(X_unlabeled, X_labeled, y_labeled, X_val, y_val):
    """
    Semi-supervised learning using cluster-based label propagation.

    We:
      1. Cluster labeled + unlabeled samples.
      2. Assign each cluster a label using the nearest labeled example.
      3. Train a nonlinear classifier on the resulting pseudo-labeled data.
      4. Tune the number of clusters / SVM regularization on the validation set.
    """

    X_unlabeled = np.asarray(X_unlabeled)
    X_labeled = np.asarray(X_labeled)
    y_labeled = np.asarray(y_labeled, dtype=np.int64)
    X_val = np.asarray(X_val)
    y_val = np.asarray(y_val, dtype=np.int64)

    X_all = np.concatenate([X_labeled, X_unlabeled], axis=0)
    n_labeled = len(X_labeled)

    # A small set of cluster counts works well for digits while staying fast.
    candidates = [20, 30, 40, 50, 60]

    best_acc = -1.0
    best_pseudo = None
    best_kmeans = None
    best_svc_C = None

    # Tune cluster count and downstream classifier on validation data.
    for n_clusters in candidates:
        kmeans = KMeans(
            n_clusters=n_clusters,
            n_init=10,
            random_state=42,
            max_iter=300,
        )

        cluster_ids = kmeans.fit_predict(X_all)
        centers = kmeans.cluster_centers_

        # Assign every cluster the label of its nearest labeled sample.
        # This is more robust than assuming every cluster contains a
        # labeled example.
        dists = np.sum(
            (centers[:, None, :] - X_labeled[None, :, :]) ** 2,
            axis=2,
        )
        nearest_labeled = np.argmin(dists, axis=1)
        cluster_labels = y_labeled[nearest_labeled]

        # Preserve the true labels of the original labeled examples.
        pseudo_y = cluster_labels[cluster_ids].astype(np.int64)
        pseudo_y[:n_labeled] = y_labeled

        for C in (1.0, 3.0, 10.0):
            clf = SVC(
                C=C,
                kernel="rbf",
                gamma="scale",
                random_state=42,
            )
            clf.fit(X_all, pseudo_y)

            pred = clf.predict(X_val)
            acc = accuracy_score(y_val, pred)

            if acc > best_acc:
                best_acc = acc
                best_pseudo = pseudo_y.copy()
                best_kmeans = kmeans
                best_svc_C = C

    # Retrain the final downstream classifier using the best pseudo-labels.
    final_clf = SVC(
        C=best_svc_C,
        kernel="rbf",
        gamma="scale",
        random_state=42,
    )
    final_clf.fit(X_all, best_pseudo)

    def predict(X):
        X = np.asarray(X)
        return final_clf.predict(X).astype(np.int64)

    return predict