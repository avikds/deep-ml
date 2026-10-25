import math
from collections import Counter


def train(text: str):
    """Train an interpolated character n-gram language model."""
    max_order = 5
    alpha = 0.1

    unigram = Counter(text)
    vocab = set(text)

    # counts[n][context] stores counts of the next character.
    counts = [None] + [{} for _ in range(max_order)]

    for i, ch in enumerate(text):
        for n in range(1, min(max_order, i) + 1):
            context = text[i - n:i]
            table = counts[n]

            if context not in table:
                table[context] = Counter()
            table[context][ch] += 1

    # Precompute context totals and number of distinct continuations.
    for n in range(1, max_order + 1):
        for context, counter in counts[n].items():
            total = sum(counter.values())
            distinct = len(counter)
            counts[n][context] = (counter, total, distinct)

    return {
        "max_order": max_order,
        "alpha": alpha,
        "unigram": unigram,
        "vocab": vocab,
        "vocab_size": len(vocab),
        "train_length": len(text),
        "counts": counts,
    }


def logprob(model, seq: str) -> list[float]:
    """Return causal natural-log probabilities, one per character."""
    max_order = model["max_order"]
    alpha = model["alpha"]
    unigram = model["unigram"]
    counts = model["counts"]

    # One extra category reserves probability for unseen characters.
    denominator = (
        model["train_length"]
        + alpha * (model["vocab_size"] + 1)
    )

    result = []

    for i, ch in enumerate(seq):
        # Smoothed unconditional character probability.
        p = (unigram.get(ch, 0) + alpha) / denominator

        # Interpolate successively with higher-order contexts.
        for n in range(1, min(max_order, i) + 1):
            context = seq[i - n:i]
            entry = counts[n].get(context)

            if entry is None:
                continue

            counter, total, distinct = entry

            # Witten-Bell interpolation:
            # observed counts plus backed-off lower-order probability.
            p = (
                counter.get(ch, 0) + distinct * p
            ) / (total + distinct)

        # Ensure a finite, non-positive log probability.
        result.append(math.log(max(min(p, 1.0), 1e-300)))

    return result