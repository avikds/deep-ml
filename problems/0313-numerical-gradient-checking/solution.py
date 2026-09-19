import numpy as np

def numerical_gradient_check(f, x, analytical_grad, epsilon=1e-7):
    """
    Perform numerical gradient checking using centered finite differences.
    """
    x = np.asarray(x, dtype=float)
    analytical_grad = np.asarray(analytical_grad, dtype=float)

    numerical_grad = np.zeros_like(x, dtype=float)

    for idx in np.ndindex(x.shape):
        original = x[idx]

        x[idx] = original + epsilon
        f_plus = f(x)

        x[idx] = original - epsilon
        f_minus = f(x)

        x[idx] = original
        numerical_grad[idx] = (f_plus - f_minus) / (2.0 * epsilon)

    numerator = np.linalg.norm(numerical_grad - analytical_grad)
    denominator = np.linalg.norm(numerical_grad) + np.linalg.norm(analytical_grad)

    if denominator == 0:
        relative_error = 0.0
    else:
        relative_error = numerator / denominator

    return numerical_grad, float(relative_error)