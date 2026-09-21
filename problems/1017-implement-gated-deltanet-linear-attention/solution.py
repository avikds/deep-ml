import numpy as np

def gated_deltanet(q, k, v, a, b, g, A_log, rms_weight, eps=1e-6):
    """
    Simplified Gated DeltaNet linear attention forward pass.
    """
    q = np.asarray(q, dtype=float)
    k = np.asarray(k, dtype=float)
    v = np.asarray(v, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    g = np.asarray(g, dtype=float)
    rms_weight = np.asarray(rms_weight, dtype=float)

    T, d = q.shape

    # L2-normalize queries and keys.
    q_norm = q / (np.linalg.norm(q, axis=1, keepdims=True) + eps)
    k_norm = k / (np.linalg.norm(k, axis=1, keepdims=True) + eps)

    # Numerically stable softplus.
    softplus_a = np.logaddexp(0.0, a)

    # Gates.
    alpha = np.exp(-softplus_a * np.exp(A_log))
    beta = 1.0 / (1.0 + np.exp(-b))

    # Recurrent matrix memory.
    S = np.zeros((d, d), dtype=float)
    outputs = np.zeros((T, d), dtype=float)

    for t in range(T):
        # 1. Decay.
        S *= alpha[t]

        # 2. Delta-rule prediction error.
        prediction = S.T @ k_norm[t]
        delta = (v[t] - prediction) * beta[t]

        # 3. Memory update.
        S += np.outer(k_norm[t], delta)

        # 4. Read out with normalized query.
        outputs[t] = S.T @ q_norm[t]

    # RMSNorm per timestep.
    rms = np.sqrt(np.mean(outputs ** 2, axis=1, keepdims=True) + eps)
    o_norm = (outputs / rms) * rms_weight

    # SiLU output gate.
    g_sigmoid = 1.0 / (1.0 + np.exp(-g))
    y = o_norm * (g * g_sigmoid)

    return np.round(y, 4).tolist()