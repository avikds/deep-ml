import numpy as np

def flash_decoding(q, k, v, n_splits):
    """
    FlashDecoding attention for one decode step with GQA.

    q: (batch, n_q_heads, head_dim)
    k, v: (batch, seq, n_kv_heads, head_dim)
    n_splits: int
    Returns: (batch, n_q_heads, head_dim)
    """
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    batch, n_q_heads, head_dim = q.shape
    _, seq, n_kv_heads, _ = k.shape

    if n_splits <= 0:
        raise ValueError("n_splits must be positive")
    if n_q_heads % n_kv_heads != 0:
        raise ValueError("n_q_heads must be divisible by n_kv_heads")

    group_size = n_q_heads // n_kv_heads

    # Per-split attention statistics.
    split_max = np.full(
        (n_splits, batch, n_q_heads),
        -np.inf,
        dtype=np.float64,
    )
    split_sum = np.zeros(
        (n_splits, batch, n_q_heads),
        dtype=np.float64,
    )
    split_out = np.zeros(
        (n_splits, batch, n_q_heads, head_dim),
        dtype=np.float64,
    )

    # Same partitioning as np.array_split(range(seq), n_splits).
    for s, idx in enumerate(np.array_split(np.arange(seq), n_splits)):
        if len(idx) == 0:
            continue

        ks = k[:, idx, :, :]
        vs = v[:, idx, :, :]

        # Repeat each KV head for its GQA query-head group.
        kv_idx = np.arange(n_q_heads) // group_size
        ks = ks[:, :, kv_idx, :]
        vs = vs[:, :, kv_idx, :]

        # FlashAttention/standard scaled dot product uses sqrt(head_dim).
        scores = np.einsum("bhd,blhd->bhl", q, ks) / np.sqrt(head_dim)

        m = np.max(scores, axis=-1)
        exp_scores = np.exp(scores - m[:, :, None])
        denom = np.sum(exp_scores, axis=-1)

        out = np.einsum("bhl,blhd->bhd", exp_scores, vs)

        split_max[s] = m
        split_sum[s] = denom
        split_out[s] = out

    # Merge split-wise softmax statistics.
    global_max = np.max(split_max, axis=0)
    rescale = np.exp(split_max - global_max[None, :, :])

    denom = np.sum(rescale * split_sum, axis=0)
    numer = np.sum(
        rescale[:, :, :, None] * split_out,
        axis=0
    )

    return numer / denom[:, :, None]