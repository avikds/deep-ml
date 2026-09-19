import numpy as np

def combined_sampling(
    logits: list[float],
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    seed: int = 42
) -> dict:
    """
    Apply temperature scaling, top-k, top-p, then sample a token.
    """
    logits = np.asarray(logits, dtype=float)
    vocab_size = len(logits)

    if vocab_size == 0:
        raise ValueError("logits must not be empty")

    if temperature <= 0:
        # Greedy decoding
        token = int(np.argmax(logits))
        probs = np.zeros(vocab_size, dtype=float)
        probs[token] = 1.0

        return {
            "probabilities": np.round(probs, 4).tolist(),
            "sampled_token": token
        }

    if not (0 < top_p <= 1.0):
        raise ValueError("top_p must be in (0, 1]")

    # 1. Temperature scaling
    scaled = logits / temperature

    # 2. Top-k filtering
    filtered = scaled.copy()

    if top_k > 0 and top_k < vocab_size:
        # Stable tie-breaking by original index.
        order = np.argsort(-filtered, kind="stable")
        keep = order[:top_k]

        mask = np.ones(vocab_size, dtype=bool)
        mask[keep] = False
        filtered[mask] = -np.inf

    # 3. Stable softmax
    max_logit = np.max(filtered)
    exp_logits = np.exp(filtered - max_logit)
    exp_logits[~np.isfinite(filtered)] = 0.0

    total = np.sum(exp_logits)
    if total <= 0:
        probs = np.zeros(vocab_size, dtype=float)
        probs[int(np.argmax(scaled))] = 1.0
    else:
        probs = exp_logits / total

    # 4. Top-p / nucleus filtering
    if top_p < 1.0:
        order = np.argsort(-probs, kind="stable")
        sorted_probs = probs[order]
        cumulative = np.cumsum(sorted_probs)

        # Keep the smallest set whose cumulative probability >= top_p.
        cutoff = np.searchsorted(cumulative, top_p, side="left")
        keep_count = cutoff + 1

        keep = order[:keep_count]

        nucleus = np.zeros(vocab_size, dtype=float)
        nucleus[keep] = probs[keep]

        total = nucleus.sum()
        if total > 0:
            probs = nucleus / total

    # 5. Sampling
    rng = np.random.default_rng(seed)
    sampled_token = int(rng.choice(vocab_size, p=probs))

    return {
        "probabilities": np.round(probs, 4).tolist(),
        "sampled_token": sampled_token
    }