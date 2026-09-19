import numpy as np

def deterministic_moe_backward(
    contributions: list,
    num_ranks: int,
    num_tokens: int,
    dim: int
) -> list:
    """
    Deterministically accumulate MoE backward-pass token gradients.
    """

    # Group contributions by source rank.
    per_rank = [[] for _ in range(num_ranks)]

    for rank, token_id, grad in contributions:
        per_rank[rank].append((token_id, np.asarray(grad, dtype=float)))

    # One isolated accumulation buffer per rank.
    buffers = np.zeros(
        (num_ranks, num_tokens, dim),
        dtype=np.float64
    )

    # Stable sort by destination token id within each rank.
    for rank in range(num_ranks):
        per_rank[rank].sort(key=lambda item: item[0])

        # Accumulate in the sorted order.
        for token_id, grad in per_rank[rank]:
            buffers[rank, token_id] += grad

    # Deterministic reduction in ascending rank order.
    result = np.zeros((num_tokens, dim), dtype=np.float64)

    for rank in range(num_ranks):
        result += buffers[rank]

    return result.tolist()