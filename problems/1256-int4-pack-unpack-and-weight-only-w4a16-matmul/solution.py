import numpy as np

def pack_int4(q: np.ndarray) -> np.ndarray:
    """Pack even-length nibble array (0..15) into uint8 bytes: lo | (hi<<4)."""
    q = np.asarray(q, dtype=np.uint8)
    assert q.ndim == 1 and len(q) % 2 == 0

    lo = q[0::2]
    hi = q[1::2]
    return lo | (hi << 4)


def unpack_int4(packed: np.ndarray, n: int) -> np.ndarray:
    """Unpack bytes to length-n int nibbles."""
    packed = np.asarray(packed, dtype=np.uint8)

    out = np.empty(n, dtype=np.uint8)
    out[0::2] = packed[:(n + 1) // 2] & 0x0F
    out[1::2] = (packed[:n // 2] >> 4) & 0x0F

    return out


def w4a16_matmul(x, packed_w, scales, in_features: int):
    """
    x: (batch, in_features) float
    packed_w: (out_features, in_features//2) uint8
    scales: (out_features,)
    Dequant: W = scale * (q - 8), y = x @ W.T
    """
    x = np.asarray(x, dtype=float)
    packed_w = np.asarray(packed_w, dtype=np.uint8)
    scales = np.asarray(scales, dtype=float)

    out_features = packed_w.shape[0]

    # Unpack each output row along the input-feature dimension.
    Wq = np.empty((out_features, in_features), dtype=np.uint8)
    Wq[:, 0::2] = packed_w & 0x0F
    Wq[:, 1::2] = packed_w >> 4

    # Per-row dequantization with zero-point 8.
    W = scales[:, None] * (Wq.astype(float) - 8.0)

    return x @ W.T