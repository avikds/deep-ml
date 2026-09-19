import numpy as np
import hashlib

def minhash_near_duplicates(
    documents: list[str],
    num_hashes: int,
    threshold: float,
    shingle_size: int,
    seed: int
) -> list[tuple]:
    """
    Detect near-duplicate document pairs using MinHash.

    Returns a list of (i, j) index pairs (i < j) whose estimated Jaccard
    similarity meets or exceeds the given threshold.
    """
    # Large Mersenne prime: 2^61 - 1
    p = 2**61 - 1

    rng = np.random.default_rng(seed)
    a = rng.integers(1, p, size=num_hashes)
    b = rng.integers(0, p, size=num_hashes)

    def shingles(document):
        tokens = document.lower().split()

        if len(tokens) == 0:
            return set()

        if len(tokens) < shingle_size:
            return {" ".join(tokens)}

        return {
            " ".join(tokens[i:i + shingle_size])
            for i in range(len(tokens) - shingle_size + 1)
        }

    def signature(document):
        shingle_set = shingles(document)

        # Empty documents get the sentinel value p.
        if not shingle_set:
            return np.full(num_hashes, p, dtype=np.int64)

        sig = np.full(num_hashes, p, dtype=np.int64)

        for shingle in shingle_set:
            base = int(
                hashlib.md5(shingle.encode("utf-8")).hexdigest(),
                16
            ) % p

            # Use Python integer arithmetic to avoid int64 multiplication
            # overflow before reducing modulo p.
            for h in range(num_hashes):
                value = (int(a[h]) * base + int(b[h])) % p
                if value < sig[h]:
                    sig[h] = value

        return sig

    signatures = [signature(doc) for doc in documents]

    result = []

    for i in range(len(documents)):
        for j in range(i + 1, len(documents)):
            similarity = float(
                np.mean(signatures[i] == signatures[j])
            )

            if similarity >= threshold:
                result.append((i, j))

    return result