from collections import Counter

def byte_pair_encoding(corpus: dict, num_merges: int) -> list:
    """
    Train a BPE tokenizer on the given corpus and return its merge operations.
    """
    words = {
        tuple(sequence.split()): freq
        for sequence, freq in corpus.items()
    }

    merges = []

    for _ in range(num_merges):
        pair_counts = Counter()

        # Count adjacent pairs, weighted by each sequence's frequency.
        for tokens, freq in words.items():
            for i in range(len(tokens) - 1):
                pair_counts[(tokens[i], tokens[i + 1])] += freq

        if not pair_counts:
            break

        # Counter preserves insertion order, so ties select the first pair.
        best_pair = max(pair_counts, key=pair_counts.get)

        if pair_counts[best_pair] <= 0:
            break

        merges.append(best_pair)
        merged_token = best_pair[0] + best_pair[1]

        # Merge all non-overlapping occurrences, left to right.
        new_words = {}

        for tokens, freq in words.items():
            result = []
            i = 0

            while i < len(tokens):
                if (
                    i < len(tokens) - 1
                    and tokens[i] == best_pair[0]
                    and tokens[i + 1] == best_pair[1]
                ):
                    result.append(merged_token)
                    i += 2
                else:
                    result.append(tokens[i])
                    i += 1

            key = tuple(result)
            new_words[key] = new_words.get(key, 0) + freq

        words = new_words

    return merges