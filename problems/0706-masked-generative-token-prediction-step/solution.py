import numpy as np
import math

def maskgit_decode_step(tokens, logits, step: int, total_steps: int, mask_id: int) -> np.ndarray:
    """
    Perform a single step of parallel masked token decoding.
    """
    tokens = np.asarray(tokens).copy()
    logits = np.asarray(logits, dtype=float)

    N = len(tokens)

    # Positions that were masked before this decoding step.
    originally_masked = (tokens == mask_id)

    # Stable softmax over vocabulary dimension.
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # Deterministic argmax predictions and their confidences.
    predictions = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)

    # Fill currently masked positions with the model predictions.
    tokens[originally_masked] = predictions[originally_masked]

    # Number of positions that should remain masked.
    remaining = int(
        math.floor(
            N * math.cos(
                math.pi / 2 * (step + 1) / total_steps
            )
        )
    )

    # Only positions that were masked before this step may be re-masked.
    candidates = np.flatnonzero(originally_masked)

    # Protect positions that were already unmasked.
    if len(candidates) > 0:
        # Lowest confidence first; np.argsort provides the required
        # deterministic tie-breaking.
        order = np.argsort(confidences[candidates])
        to_mask = candidates[order[:min(remaining, len(candidates))]]
        tokens[to_mask] = mask_id

    return tokens