import numpy as np

def quantize(W, bits):
    """
    Per-input-channel asymmetric INT4 quantization.

    Each input-feature column gets its own scale and zero point.
    """
    W = np.asarray(W, dtype=np.float32)

    if bits != 4:
        raise ValueError("This implementation expects bits=4")

    q_levels = 2 ** bits - 1  # 15

    # Quantize each input channel independently.
    w_min = np.min(W, axis=0)
    w_max = np.max(W, axis=0)

    scale = (w_max - w_min) / q_levels
    scale_safe = np.where(scale > 0, scale, 1.0)

    codes = np.round(
        (W - w_min[None, :]) / scale_safe[None, :]
    )
    codes = np.clip(codes, 0, q_levels).astype(np.uint8)

    # Degenerate constant columns decode exactly.
    scale = np.where(scale > 0, scale, 1.0)

    return {
        "codes": codes,
        "scale": scale.astype(np.float16),
        "zero": w_min.astype(np.float16),
    }


def dequantize(q):
    """Reconstruct the original weight matrix from the quantized dict."""
    codes = q["codes"].astype(np.float32)
    scale = q["scale"].astype(np.float32)
    zero = q["zero"].astype(np.float32)

    return codes * scale[None, :] + zero[None, :]