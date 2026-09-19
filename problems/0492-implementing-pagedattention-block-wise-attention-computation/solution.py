import numpy as np


def paged_attention(
    query: np.ndarray,
    key_cache: np.ndarray,
    value_cache: np.ndarray,
    block_table: list,
    context_len: int
) -> np.ndarray:
    """
    Perform scaled dot-product attention with paged KV cache.

    Args:
        query:       (num_heads, head_dim) query for a single token
        key_cache:   (num_physical_blocks, block_size, num_heads, head_dim)
        value_cache: (num_physical_blocks, block_size, num_heads, head_dim)
        block_table: list of physical block indices (logical -> physical)
        context_len: number of valid KV tokens in the sequence

    Returns:
        (num_heads, head_dim) attention output, rounded to 4 decimal places
    """

    query = np.asarray(query, dtype=float)
    key_cache = np.asarray(key_cache, dtype=float)
    value_cache = np.asarray(value_cache, dtype=float)

    num_heads, head_dim = query.shape
    block_size = key_cache.shape[1]

    if context_len <= 0:
        return np.zeros((num_heads, head_dim), dtype=float)

    # Number of logical blocks actually needed.
    num_blocks = (context_len + block_size - 1) // block_size

    keys = []
    values = []

    remaining = context_len

    # Reconstruct KV cache in logical token order.
    for logical_block in range(num_blocks):
        physical_block = block_table[logical_block]

        tokens_in_block = min(block_size, remaining)

        keys.append(key_cache[physical_block, :tokens_in_block])
        values.append(value_cache[physical_block, :tokens_in_block])

        remaining -= tokens_in_block

    # Shape:
    # keys   -> (context_len, num_heads, head_dim)
    # values -> (context_len, num_heads, head_dim)
    keys = np.concatenate(keys, axis=0)
    values = np.concatenate(values, axis=0)

    # Compute attention scores independently for each head.
    # query: (H, D)
    # keys:  (T, H, D)
    # scores -> (H, T)
    scores = np.einsum("hd,thd->ht", query, keys)
    scores *= 1.0 / np.sqrt(head_dim)

    # Numerically stable softmax over tokens.
    scores = scores - np.max(scores, axis=1, keepdims=True)
    weights = np.exp(scores)
    weights /= np.sum(weights, axis=1, keepdims=True)

    # Weighted sum of values.
    # weights: (H, T)
    # values:  (T, H, D)
    output = np.einsum("ht,thd->hd", weights, values)

    return np.round(output, 4)