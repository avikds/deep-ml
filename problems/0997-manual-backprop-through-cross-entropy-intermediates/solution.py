import numpy as np

def manual_cross_entropy_backward(logits: np.ndarray, y: np.ndarray) -> dict:
    """
    Manually compute gradients through every intermediate of softmax + NLL loss.
    """
    logits = np.asarray(logits, dtype=float)
    y = np.asarray(y, dtype=int)

    n, C = logits.shape

    # Forward intermediates.
    logit_maxes = logits.max(axis=1, keepdims=True)
    norm_logits = logits - logit_maxes
    counts = np.exp(norm_logits)
    counts_sum = counts.sum(axis=1, keepdims=True)
    counts_sum_inv = counts_sum ** -1
    probs = counts * counts_sum_inv
    logprobs = np.log(probs)

    # 1. dloss / dlogprobs
    dlogprobs = np.zeros_like(logprobs)
    dlogprobs[np.arange(n), y] = -1.0 / n

    # 2. logprobs = log(probs)
    dprobs = dlogprobs / probs

    # 3. probs = counts * counts_sum_inv
    dcounts = dprobs * counts_sum_inv
    dcounts_sum_inv = np.sum(
        dprobs * counts,
        axis=1,
        keepdims=True
    )

    # 4. counts_sum_inv = counts_sum ** -1
    dcounts_sum = (
        -counts_sum ** -2
    ) * dcounts_sum_inv

    # 5. counts_sum = sum(counts)
    dcounts += dcounts_sum * np.ones_like(counts)

    # 6. counts = exp(norm_logits)
    dnorm_logits = counts * dcounts

    # 7. norm_logits = logits - logit_maxes
    dlogits = dnorm_logits.copy()
    dlogit_maxes = -np.sum(
        dnorm_logits,
        axis=1,
        keepdims=True
    )

    # 8. logit_maxes = max(logits)
    max_indices = np.argmax(logits, axis=1)
    dlogits[np.arange(n), max_indices] += dlogit_maxes[:, 0]

    return {
        "dlogprobs": dlogprobs,
        "dprobs": dprobs,
        "dcounts_sum_inv": dcounts_sum_inv,
        "dcounts_sum": dcounts_sum,
        "dcounts": dcounts,
        "dnorm_logits": dnorm_logits,
        "dlogit_maxes": dlogit_maxes,
        "dlogits": dlogits,
    }