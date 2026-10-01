import numpy as np

def agentopsd_turn_advantages(
    turn_evidence,
    A_seq,
    B0,
    gamma: float = 0.95,
    b: float = 0.2,
    lam: float = 0.5,
    eps: float = 1e-4,
):
    """Full AgentOPSD credit assignment → (K,) reshaped advantages."""
    e = np.asarray(turn_evidence, dtype=np.float64)
    K = e.shape[0]

    # Prior belief -> logit, with clipping for numerical safety.
    B0 = float(np.clip(B0, eps, 1.0 - eps))
    l0 = np.log(B0 / (1.0 - B0))

    # Stable sigmoid.
    def sigmoid(x):
        x = np.asarray(x, dtype=np.float64)
        out = np.empty_like(x)
        pos = x >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        exp_x = np.exp(x[~pos])
        out[~pos] = exp_x / (1.0 + exp_x)
        return out

    # Evidence accumulation and belief trajectory.
    c = 0.0
    B_prev = B0
    deltas = np.empty(K, dtype=np.float64)

    for k in range(K):
        c = gamma * c + e[k]
        B_k = float(sigmoid(l0 + c))
        deltas[k] = B_k - B_prev
        B_prev = B_k

    # Sign-adjusted credit.
    q = np.sign(float(A_seq)) * deltas

    # Within-trajectory population z-score.
    mean_q = np.mean(q)
    std_q = np.std(q)
    z = (q - mean_q) / (std_q + eps)

    # Bounded reshaping multiplier.
    w = np.clip(1.0 + b * z, 1.0 - b, 1.0 + b)

    # Final turn-level advantages.
    A_tilde = float(A_seq) * ((1.0 - lam) + lam * w)

    return np.asarray(A_tilde, dtype=np.float64).reshape(K)