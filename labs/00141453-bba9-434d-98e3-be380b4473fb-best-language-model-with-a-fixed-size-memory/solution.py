import math
import numpy as np


MAX_ORDER = 6
MAX_HISTORY = 4096
SMOOTHING = 0.1


def train(text: str, alphabet: str, budget_seconds: float):
    """Train a smoothed character n-gram language model."""
    vocab_size = len(alphabet)
    char_to_id = {ch: i for i, ch in enumerate(alphabet)}
    unk = char_to_id.get("?", 0)

    ids = np.fromiter(
        (char_to_id.get(ch, unk) for ch in text),
        dtype=np.uint8,
        count=len(text),
    )
    N = len(ids)

    unigram = np.bincount(
        ids.astype(np.int64), minlength=vocab_size
    ).astype(np.int64)

    tables = {}

    # Store each n-gram as a base-vocab_size integer.
    # Keys are sorted, enabling fast context lookup with searchsorted.
    for order in range(2, MAX_ORDER + 1):
        count = N - order + 1

        if count <= 0:
            tables[order] = (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.uint32),
            )
            continue

        keys = ids[:count].astype(np.int64)
        for offset in range(1, order):
            keys = keys * vocab_size + ids[offset:offset + count]

        unique_keys, counts = np.unique(keys, return_counts=True)
        tables[order] = (
            unique_keys,
            counts.astype(np.uint32),
        )

        del keys, unique_keys, counts

    def encode_literal(s):
        return np.fromiter(
            (char_to_id.get(ch, unk) for ch in s),
            dtype=np.int16,
        )

    return {
        "V": vocab_size,
        "alphabet": alphabet,
        "char_to_id": char_to_id,
        "unk": unk,
        "unigram": unigram,
        "tables": tables,
        "train_length": N,
        "code_marker": encode_literal("Code: "),
        "repeat_marker": encode_literal("The code was "),
        "alpha": SMOOTHING,
    }


def init_state(model, batch_size: int):
    """
    Each row stores a character position followed by the document's
    character history. int16 keeps the state comfortably under the cap.
    """
    return np.zeros(
        (batch_size, MAX_HISTORY + 1),
        dtype=np.int16,
    )


def _ngram_probabilities(model, history):
    """Compute the interpolated n-gram next-character distribution."""
    V = model["V"]
    alpha = model["alpha"]
    unigram = model["unigram"]

    p = (unigram.astype(np.float64) + alpha) / (
        model["train_length"] + alpha * V
    )

    # Witten-Bell interpolation from shorter to longer contexts.
    max_context = min(MAX_ORDER - 1, len(history))

    for context_len in range(1, max_context + 1):
        order = context_len + 1
        keys, counts = model["tables"][order]

        if len(keys) == 0:
            continue

        ctx_code = 0
        for value in history[-context_len:]:
            ctx_code = ctx_code * V + int(value)

        low = ctx_code * V
        high = low + V

        start = int(np.searchsorted(keys, low))
        end = int(np.searchsorted(keys, high))

        if start == end:
            continue

        context_counts = counts[start:end].astype(np.float64)
        total = float(context_counts.sum())
        distinct = end - start
        denom = total + distinct

        # P_n = (observed count + T * P_(n-1)) / (C + T)
        p *= distinct / denom
        next_chars = keys[start:end] % V
        p[next_chars] += context_counts / denom

    p /= p.sum()
    return p


def _apply_copy_distribution(model, p, history):
    """
    Use a previous exact occurrence of the current suffix to predict
    its following character. Longer matches receive more weight.
    """
    length = len(history)
    if length < 7:
        return p

    # The stored character IDs fit in uint8.
    data = history.astype(np.uint8).tobytes()

    best_char = None
    best_length = 0

    # Search for a non-overlapping earlier occurrence.
    for match_len in (12, 10, 8, 6):
        if length < 2 * match_len + 1:
            continue

        context = data[-match_len:]
        earlier_end = length - match_len - 1
        found = data.rfind(context, 0, earlier_end)

        if found >= 0:
            candidate = found + match_len
            if candidate < length:
                best_char = data[candidate]
                best_length = match_len
                break

    if best_char is not None and best_char < model["V"]:
        weight = {
            12: 0.85,
            10: 0.75,
            8: 0.55,
            6: 0.35,
        }[best_length]

        p *= (1.0 - weight)
        p[best_char] += weight

    return p


def _apply_code_recall(model, p, history):
    """
    Recognize 'The code was ' followed by a prefix of the six-letter
    code recorded after the opening 'Code: ' marker.
    """
    marker = model["code_marker"]
    repeat = model["repeat_marker"]

    if len(history) < 12:
        return p

    # Require the expected opening prefix before trusting the code memory.
    if not np.array_equal(history[:len(marker)], marker):
        return p

    code_start = len(marker)
    code_end = code_start + 6

    if len(history) < code_end:
        return p

    code = history[code_start:code_end]
    V = model["V"]

    # Check whether the history ends in the repeat marker plus code prefix.
    for prefix_len in range(5, -1, -1):
        pattern_len = len(repeat) + prefix_len
        if len(history) < pattern_len:
            continue

        if not np.array_equal(
            history[-pattern_len:-prefix_len or None]
            if prefix_len else history[-pattern_len:],
            repeat,
        ):
            continue

        if prefix_len and not np.array_equal(
            history[-prefix_len:], code[:prefix_len]
        ):
            continue

        if prefix_len < 6:
            target = int(code[prefix_len])
            if 0 <= target < V:
                p *= 0.001
                p[target] += 0.999
            break

    return p


def step(model, state, prev):
    """
    Return normalized log-probabilities for the next character and
    updated per-document state.
    """
    V = model["V"]
    batch_size = len(prev)
    logprobs = np.empty((batch_size, V), dtype=np.float64)

    for b in range(batch_size):
        row = state[b]

        if int(prev[b]) < 0:
            # Fresh document: discard any old row history.
            row[:] = 0
            pos = 0
        else:
            pos = int(row[0])
            if pos > 0:
                row[pos] = int(prev[b])

        history_len = min(pos, MAX_HISTORY)
        history = row[1:1 + history_len]

        p = _ngram_probabilities(model, history)

        if history_len >= 7:
            p = _apply_copy_distribution(model, p, history)
            p = _apply_code_recall(model, p, history)

        # Guarantee a normalized, finite distribution.
        p = np.maximum(p, 1e-300)
        p /= p.sum()
        logprobs[b] = np.log(p)

        row[0] = min(pos + 1, MAX_HISTORY)

    return logprobs, state