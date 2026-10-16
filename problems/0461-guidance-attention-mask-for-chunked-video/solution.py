import numpy as np

def guidance_attention_mask(
    chunk_sizes: list[int],
    current_chunk: int
) -> np.ndarray:
    """
    Build a boolean attention mask for chunked autoregressive video generation.
    """
    if not 0 <= current_chunk < len(chunk_sizes):
        raise ValueError("current_chunk is out of range")

    total_tokens = sum(chunk_sizes)
    history_end = sum(chunk_sizes[:current_chunk])
    current_end = history_end + chunk_sizes[current_chunk]

    mask = np.zeros((total_tokens, total_tokens), dtype=bool)

    # Historical tokens attend bidirectionally to all historical tokens.
    mask[:history_end, :history_end] = True

    # Current-chunk tokens attend to all history.
    mask[history_end:current_end, :history_end] = True

    # Within the current chunk, attention is causal (including self).
    current_size = chunk_sizes[current_chunk]
    mask[
        history_end:current_end,
        history_end:current_end
    ] = np.tril(np.ones((current_size, current_size), dtype=bool))

    # Future chunks remain masked until they become current or historical.
    return mask