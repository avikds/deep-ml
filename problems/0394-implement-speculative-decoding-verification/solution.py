import numpy as np

def speculative_decode_verify(
    draft_tokens: list,
    draft_probs: list,
    target_probs: list,
    coin_flips: list,
    resample_coin: float
) -> list:
    """
    Verify draft tokens using speculative decoding.
    """
    draft_tokens = list(draft_tokens)
    draft_probs = np.asarray(draft_probs, dtype=float)
    target_probs = np.asarray(target_probs, dtype=float)

    result = []

    for i, token in enumerate(draft_tokens):
        q = draft_probs[i, token]
        p = target_probs[i, token]

        # Acceptance probability: min(1, p/q)
        if q <= 0:
            accept_prob = 1.0 if p > 0 else 0.0
        else:
            accept_prob = min(1.0, p / q)

        if coin_flips[i] < accept_prob:
            result.append(int(token))
            continue

        # Rejection: residual distribution max(0, p - q)
        adjusted = np.maximum(
            target_probs[i] - draft_probs[i],
            0.0
        )

        total = np.sum(adjusted)

        if total <= 0:
            # Degenerate case: fall back to target distribution
            adjusted = np.maximum(target_probs[i], 0.0)
            total = np.sum(adjusted)

        if total <= 0:
            # Final deterministic fallback
            result.append(int(np.argmax(target_probs[i])))
            break

        adjusted = adjusted / total

        cumsum = np.cumsum(adjusted)
        token_idx = int(np.searchsorted(cumsum, resample_coin, side="left"))

        # Guard against floating-point edge cases
        token_idx = min(token_idx, len(adjusted) - 1)

        result.append(token_idx)
        break

    return result