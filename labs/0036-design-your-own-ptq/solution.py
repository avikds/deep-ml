import numpy as np


def _quantize_per_row(W):
    """
    8-bit asymmetric per-row uniform quantization.
    Returns the dequantized weights.
    """
    W = np.asarray(W, dtype=np.float64)

    w_min = np.min(W, axis=1, keepdims=True)
    w_max = np.max(W, axis=1, keepdims=True)

    scale = (w_max - w_min) / 255.0

    # Handle rows whose values are all identical.
    scale = np.where(scale > 1e-12, scale, 1.0)

    zero_point = np.round(-w_min / scale)
    zero_point = np.clip(zero_point, 0, 255)

    q = np.round(W / scale + zero_point)
    q = np.clip(q, 0, 255)

    return (q - zero_point) * scale


def _quantize_tensor(x):
    """
    8-bit asymmetric per-tensor uniform quantization.
    Returns the dequantized tensor.
    """
    x = np.asarray(x, dtype=np.float64)

    x_min = np.min(x)
    x_max = np.max(x)

    if x_max - x_min < 1e-12:
        return np.full_like(x, x_min, dtype=np.float64)

    scale = (x_max - x_min) / 255.0
    zero_point = np.round(-x_min / scale)
    zero_point = np.clip(zero_point, 0, 255)

    q = np.round(x / scale + zero_point)
    q = np.clip(q, 0, 255)

    return (q - zero_point) * scale


def ptq(weights, calib_X):
    """
    8-bit post-training quantization for the MNIST MLP.

    Weight matrices use per-row asymmetric quantization.
    Bias vectors use per-tensor asymmetric quantization.

    calib_X is accepted as required by the interface. The quantization
    itself does not need activation calibration because per-row weight
    quantization already gives a very fine 8-bit reconstruction.
    """
    dequant_weights = {}

    # Per-output-neuron quantization for weight matrices.
    for key in ("W1", "W2", "W3"):
        dequant_weights[key] = _quantize_per_row(weights[key]).astype(
            np.float32
        )

    # Quantize biases as their own tensors.
    for key in ("b1", "b2", "b3"):
        dequant_weights[key] = _quantize_tensor(weights[key]).astype(
            np.float32
        )

    return dequant_weights