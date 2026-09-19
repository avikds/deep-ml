import numpy as np

def speculative_decode(
    draft_probs: list,
    target_probs: list,
    draft_tokens: list,
    accept_rand: list,
    sample_rand: list
) -> list:
    """
    Simulate speculative decoding end-to-end.
    """
    K = len(draft_tokens)
    result = []

    def sample_from_probs(probs, u):
        probs = np.asarray(probs, dtype=float)
        probs = np.maximum(probs, 0.0)

        total = probs.sum()
        if total <= 0:
            return int(np.argmax(probs))

        probs = probs / total
        cdf = np.cumsum(probs)

        # Smallest index with CDF >= u
        idx = np.searchsorted(cdf, u, side="left")
        return int(min(idx, len(probs) - 1))

    for i in range(K):
        token = int(draft_tokens[i])

        q = float(draft_probs[i][token])
        p = float(target_probs[i][token])

        # Acceptance probability = min(1, p / q)
        if q <= 0:
            accept_prob = 1.0 if p > 0 else 0.0
        else:
            accept_prob = min(1.0, p / q)

        if accept_rand[i] < accept_prob:
            result.append(token)
        else:
            # Rejection: corrected/residual distribution
            adjusted = np.maximum(
                np.asarray(target_probs[i], dtype=float)
                - np.asarray(draft_probs[i], dtype=float),
                0.0
            )

            total = adjusted.sum()

            if total > 0:
                adjusted /= total
            else:
                # Degenerate fallback to target distribution
                adjusted = np.maximum(
                    np.asarray(target_probs[i], dtype=float),
                    0.0
                )
                total = adjusted.sum()

                if total > 0:
                    adjusted /= total
                else:
                    adjusted = np.ones_like(adjusted) / len(adjusted)

            result.append(
                sample_from_probs(adjusted, sample_rand[i])
            )

            return result

    # All draft tokens accepted: sample bonus token from target at K
    if len(target_probs) > K:
        result.append(
            sample_from_probs(
                target_probs[K],
                sample_rand[K]
            )
        )

    return result