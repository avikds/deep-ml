def train_tokenizer(corpus, vocab_size):
    """
    Train a deterministic byte-pair-encoding-style tokenizer using only
    the Python standard library.
    """
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")

    corpus = list(corpus)

    # Character vocabulary, deterministic ordering.
    chars = sorted(set("".join(corpus)))

    if len(chars) > vocab_size:
        raise ValueError("vocab_size is smaller than the character vocabulary")

    # Token id -> decoded string.
    id_to_token = list(chars)
    token_to_id = {ch: i for i, ch in enumerate(chars)}

    # Represent each training document as a list of string tokens.
    sequences = [list(text) for text in corpus]

    # Each merge is (left_token, right_token).
    merges = []

    # Learn BPE merges until vocabulary is full.
    while len(id_to_token) < vocab_size:
        pair_counts = {}

        for seq in sequences:
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

        if not pair_counts:
            break

        # Highest frequency first; lexicographic tie-break for determinism.
        best_pair = min(
            pair_counts,
            key=lambda p: (-pair_counts[p], p[0], p[1])
        )

        left, right = best_pair
        merged = left + right

        # Avoid pathological duplicate tokens.
        if merged in token_to_id:
            break

        token_to_id[merged] = len(id_to_token)
        id_to_token.append(merged)
        merges.append(best_pair)

        # Apply the merge to every training sequence.
        new_sequences = []

        for seq in sequences:
            new_seq = []
            i = 0

            while i < len(seq):
                if (
                    i + 1 < len(seq)
                    and seq[i] == left
                    and seq[i + 1] == right
                ):
                    new_seq.append(merged)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1

            new_sequences.append(new_seq)

        sequences = new_sequences

    # Give each merge its learned priority.
    merge_rank = {pair: i for i, pair in enumerate(merges)}

    def encode(text):
        if not text:
            return []

        tokens = list(text)

        # Apply merges in learned order, exactly as during training.
        for left, right in merges:
            merged = left + right
            out = []
            i = 0

            while i < len(tokens):
                if (
                    i + 1 < len(tokens)
                    and tokens[i] == left
                    and tokens[i + 1] == right
                ):
                    out.append(merged)
                    i += 2
                else:
                    out.append(tokens[i])
                    i += 1

            tokens = out

        return [token_to_id[t] for t in tokens]

    def decode(token_ids):
        parts = []

        for idx in token_ids:
            idx = int(idx)
            if idx < 0 or idx >= len(id_to_token):
                raise ValueError("Invalid token id")
            parts.append(id_to_token[idx])

        return "".join(parts)

    return encode, decode