import numpy as np
from collections import Counter

def bleu_score(candidate: list[str], references: list[list[str]], max_n: int = 4) -> float:
    """
    Calculate BLEU score for a candidate sentence against reference sentences.
    """
    if not candidate or not references or max_n < 1:
        return 0.0

    candidate_len = len(candidate)

    # Choose reference length closest to the candidate length;
    # break ties by choosing the shorter reference.
    ref_len = min(
        (len(ref) for ref in references),
        key=lambda length: (abs(length - candidate_len), length)
    )

    precisions = []

    for n in range(1, max_n + 1):
        candidate_ngrams = Counter(
            tuple(candidate[i:i + n])
            for i in range(candidate_len - n + 1)
        )

        if not candidate_ngrams:
            return 0.0

        # Maximum reference count for each n-gram.
        max_ref_counts = Counter()
        for ref in references:
            ref_counts = Counter(
                tuple(ref[i:i + n])
                for i in range(len(ref) - n + 1)
            )
            for ngram, count in ref_counts.items():
                max_ref_counts[ngram] = max(
                    max_ref_counts[ngram], count
                )

        # Modified n-gram precision with clipped counts.
        clipped_count = sum(
            min(count, max_ref_counts[ngram])
            for ngram, count in candidate_ngrams.items()
        )
        total_count = sum(candidate_ngrams.values())
        precision = clipped_count / total_count

        if precision == 0.0:
            return 0.0

        precisions.append(precision)

    # Brevity penalty.
    if candidate_len >= ref_len:
        brevity_penalty = 1.0
    else:
        brevity_penalty = np.exp(1.0 - ref_len / candidate_len)

    # Geometric mean of modified n-gram precisions.
    geometric_mean = np.exp(np.mean(np.log(precisions)))

    return float(brevity_penalty * geometric_mean)