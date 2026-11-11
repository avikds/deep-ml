from collections import Counter
import math

K = 4
ALPHA = 1.5
BOS = 26
EOS = 27


def train(names: list[str]):
    counts = [{} for _ in range(K + 1)]
    totals = [{} for _ in range(K + 1)]

    unigram = Counter()

    for name in names:
        x = [ord(c) - 97 for c in name]
        n = len(x)

        unigram.update(x)

        # Padding on both sides.
        seq = [BOS] * K + x + [EOS] * K

        for i in range(n):
            pos = K + i
            target = x[i]

            for k in range(1, K + 1):
                left = tuple(seq[pos-k:pos])
                right = tuple(seq[pos+1:pos+1+k])

                key = (left, right)

                d = counts[k]
                t = totals[k]

                if key not in d:
                    d[key] = Counter()
                    t[key] = 0

                d[key][target] += 1
                t[key] += 1

    total = sum(unigram.values())
    base = [(unigram[i] + 0.5) / (total + 13.0)
            for i in range(26)]

    return {
        "counts": counts,
        "totals": totals,
        "base": base,
        "alpha": ALPHA,
        "K": K
    }


def predict(model, masked: str) -> list[float]:
    kmax = model["K"]
    counts = model["counts"]
    totals = model["totals"]
    alpha = model["alpha"]
    base = model["base"]

    gap = masked.index("_")
    n = len(masked)

    # Encode the visible letters.
    x = [
        BOS if i < 0 else
        EOS if i >= n else
        (0 if masked[i] == "_" else ord(masked[i]) - 97)
        for i in range(-kmax, n + kmax)
    ]

    pos = kmax + gap

    left = x[pos-kmax:pos]
    right = x[pos+1:pos+1+kmax]

    scores = []

    for c in range(26):
        p = base[c]

        # Hierarchical interpolation/backoff.
        for k in range(1, kmax + 1):
            key = (
                tuple(left[kmax-k:]),
                tuple(right[:k])
            )

            hist = counts[k].get(key)

            if hist is not None:
                total = totals[k][key]
                p = (hist.get(c, 0) + alpha * p) / (
                    total + alpha
                )

        scores.append(p)

    # Mild calibration and smoothing.
    temperature = 0.90
    maximum = max(math.log(max(p, 1e-30)) for p in scores)

    weights = [
        math.exp(temperature * (
            math.log(max(p, 1e-30)) - maximum
        ))
        for p in scores
    ]

    return [
        w + 0.001 * sum(weights) / 26
        for w in weights
    ]