import numpy as np

def activation(x):
    x = np.asarray(x)

    # Leaky ReLU: keeps strong gradient for negative activations
    result = np.where(x > 0, x, 0.01 * x)

    return result