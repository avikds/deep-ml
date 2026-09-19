import math
import random

def train(X_train, y_train, X_val, y_val, n_classes):
    """
    Train a digit classifier using only Python and math.

    Uses 3-nearest-neighbor classification with inverse-distance
    weighted voting.
    """

    # Keep local references for faster prediction.
    train_x = X_train
    train_y = y_train
    n_train = len(train_x)

    # Small epsilon prevents division by zero for an exact match.
    eps = 1e-12

    def predict(X):
        predictions = []

        for x in X:
            # Best three distances and corresponding labels.
            d1 = float("inf")
            d2 = float("inf")
            d3 = float("inf")

            y1 = 0
            y2 = 0
            y3 = 0

            # Find the 3 nearest training examples.
            for i in range(n_train):
                row = train_x[i]

                d = 0.0

                # Squared Euclidean distance.
                for j in range(64):
                    diff = x[j] - row[j]
                    d += diff * diff

                    # Distance cannot become better once it exceeds
                    # our current third-best distance.
                    if d >= d3:
                        break

                if d < d1:
                    d3, y3 = d2, y2
                    d2, y2 = d1, y1
                    d1, y1 = d, train_y[i]

                elif d < d2:
                    d3, y3 = d2, y2
                    d2, y2 = d, train_y[i]

                elif d < d3:
                    d3, y3 = d, train_y[i]

            # Distance-weighted voting.
            scores = [0.0] * n_classes

            scores[y1] += 1.0 / (d1 + eps)
            scores[y2] += 1.0 / (d2 + eps)
            scores[y3] += 1.0 / (d3 + eps)

            best_class = 0
            best_score = scores[0]

            for c in range(1, n_classes):
                if scores[c] > best_score:
                    best_score = scores[c]
                    best_class = c

            predictions.append(int(best_class))

        return predictions

    return predict