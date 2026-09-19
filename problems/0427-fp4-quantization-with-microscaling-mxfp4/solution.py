import numpy as np

def mxfp4_quantize(x: list, block_size: int = 4) -> dict:
    """
    Perform MXFP4 quantization with per-block microscaling.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    x = np.asarray(x, dtype=float).reshape(-1)
    n = len(x)

    fp4_values = np.array(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=float
    )

    quantized = []
    scales = []

    for start in range(0, n, block_size):
        block = x[start:start + block_size]
        amax = np.max(np.abs(block)) if len(block) else 0.0

        if amax == 0.0:
            scale = 1.0
        else:
            # Smallest power of 2 >= amax / 6
            raw_scale = amax / 6.0
            scale = 2.0 ** np.ceil(np.log2(raw_scale))

        scales.append(scale)

        for value in block:
            scaled = value / scale
            sign = -1.0 if scaled < 0 else 1.0
            magnitude = abs(scaled)

            # Nearest FP4 magnitude. np.argmin chooses the first value
            # on an exact tie, giving the smaller magnitude.
            idx = np.argmin(np.abs(fp4_values - magnitude))
            qmag = fp4_values[idx]

            qvalue = sign * qmag
            quantized.append(qvalue * scale)

    return {
        "quantized": [round(float(v), 4) for v in quantized[:n]],
        "scales": [round(float(s), 4) for s in scales]
    }