import numpy as np

def calculate_auc(y_true, y_scores):
    """Calculate ROC AUC; return 0.0 if all labels belong to one class."""
    y_true = np.asarray(y_true, dtype=int)
    y_scores = np.asarray(y_scores, dtype=float)

    positives = np.sum(y_true == 1)
    negatives = np.sum(y_true == 0)

    if positives == 0 or negatives == 0:
        return 0.0

    order = np.argsort(-y_scores, kind="mergesort")
    labels = y_true[order]
    scores = y_scores[order]

    tp = 0
    fp = 0
    tpr = [0.0]
    fpr = [0.0]

    i = 0
    n = len(labels)

    # Process equal scores together to handle ties correctly.
    while i < n:
        j = i
        while j < n and scores[j] == scores[i]:
            j += 1

        tp += int(np.sum(labels[i:j] == 1))
        fp += int(np.sum(labels[i:j] == 0))

        tpr.append(tp / positives)
        fpr.append(fp / negatives)
        i = j

    # Manual trapezoidal integration (avoids np.trapz).
    auc = 0.0
    for i in range(len(fpr) - 1):
        auc += (fpr[i + 1] - fpr[i]) * (tpr[i] + tpr[i + 1]) / 2.0

    return float(auc)