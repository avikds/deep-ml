import math

def single_neuron_model(
    features: list[list[float]],
    labels: list[int],
    weights: list[float],
    bias: float
) -> tuple[list[float], float]:

    probabilities = []

    for x in features:
        z = sum(w * xi for w, xi in zip(weights, x)) + bias
        p = 1.0 / (1.0 + math.exp(-z))
        probabilities.append(round(p, 4))

    mse = sum(
        (p - y) ** 2
        for p, y in zip(probabilities, labels)
    ) / len(labels)

    mse = round(mse, 4)

    return probabilities, mse