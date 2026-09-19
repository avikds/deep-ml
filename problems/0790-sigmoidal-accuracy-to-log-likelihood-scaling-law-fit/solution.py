import numpy as np

def fit_sigmoid_scaling(nll_list, acc_list, predict_nll):
    """
    Fit acc = 1 / (1 + exp(a*nll + b)) via least squares in logit space
    and predict accuracy at predict_nll.
    """
    nll = np.asarray(nll_list, dtype=float)
    acc = np.asarray(acc_list, dtype=float)

    if len(nll) != len(acc) or len(nll) == 0:
        raise ValueError("nll_list and acc_list must have the same non-zero length")

    if np.any((acc <= 0) | (acc >= 1)):
        raise ValueError("All accuracy values must satisfy 0 < acc < 1")

    # z = log((1 - acc) / acc) = a*nll + b
    z = np.log((1.0 - acc) / acc)

    A = np.column_stack((nll, np.ones_like(nll)))
    a, b = np.linalg.lstsq(A, z, rcond=None)[0]

    # Stable sigmoid
    t = float(a * predict_nll + b)
    if t >= 0:
        predicted_acc = np.exp(-t) / (1.0 + np.exp(-t))
    else:
        predicted_acc = 1.0 / (1.0 + np.exp(t))

    return [float(a), float(b), float(predicted_acc)]