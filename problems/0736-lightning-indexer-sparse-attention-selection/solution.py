import numpy as np

def lightning_indexer_topk(H_q, K_comp, W_down, W_up, w_head, k):
    """
    Select top-k compressed key-value entries for each query token.
    """
    H_q = np.asarray(H_q, dtype=float)
    K_comp = np.asarray(K_comp, dtype=float)
    W_down = np.asarray(W_down, dtype=float)
    W_up = np.asarray(W_up, dtype=float)
    w_head = np.asarray(w_head, dtype=float)

    n_q = H_q.shape[0]
    n_kv = K_comp.shape[0]
    h = len(w_head)
    c = K_comp.shape[1]

    top_k = min(k, n_kv)

    if top_k <= 0:
        return [[] for _ in range(n_q)]

    # 1. Low-rank query projection: d -> r -> h*c.
    q_proj = H_q @ W_down
    q_proj = q_proj @ W_up

    # Reshape to (n_q, h, c).
    q_heads = q_proj.reshape(n_q, h, c)

    # 2. Per-head dot products with compressed keys, then ReLU.
    # Shape: (n_q, h, n_kv)
    head_scores = np.einsum("qhc,kc->qhk", q_heads, K_comp)
    head_scores = np.maximum(head_scores, 0.0)

    # 3. Weighted sum across heads.
    combined_scores = np.sum(
        head_scores * w_head[None, :, None],
        axis=1
    )

    # 4. Descending score; stable sort preserves smaller index on ties.
    order = np.argsort(
        -combined_scores,
        axis=1,
        kind="stable"
    )

    return order[:, :top_k].astype(int).tolist()