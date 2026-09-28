import numpy as np

def simplified_gptq(
    w: np.ndarray,
    H_diag: np.ndarray,
    bits: int = 4,
    blocksize: int = 128
):
    """
    Simplified GPTQ: sequential symmetric quant with within-block
    diagonal-Hessian error compensation.
    """
    w = np.asarray(w, dtype=float)
    H_diag = np.asarray(H_diag, dtype=float)

    qmax = 2 ** (bits - 1) - 1
    max_abs = np.max(np.abs(w))
    scale = max_abs / qmax if max_abs != 0 else 1.0

    w_b = w.copy()
    q = np.zeros(len(w), dtype=int)
    w_hat = np.zeros(len(w), dtype=float)

    for start in range(0, len(w), blocksize):
        end = min(start + blocksize, len(w))

        for i in range(start, end):
            qi = np.round(w_b[i] / scale)
            qi = np.clip(qi, -qmax, qmax)
            q[i] = int(qi)

            w_hat[i] = q[i] * scale
            e = w_b[i] - w_hat[i]

            for j in range(i + 1, end):
                w_b[j] -= (
                    e * H_diag[i] / (H_diag[j] + 1e-8)
                )

    return q, float(scale), w_hat