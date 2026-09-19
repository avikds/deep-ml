import numpy as np

def multi_head_attention_einsum(X, W_q, W_k, W_v, W_o, num_heads: int, causal: bool = True):
    """
    Multi-head self-attention computed with numpy.einsum.
    """
    X = np.asarray(X, dtype=float)
    W_q = np.asarray(W_q, dtype=float)
    W_k = np.asarray(W_k, dtype=float)
    W_v = np.asarray(W_v, dtype=float)
    W_o = np.asarray(W_o, dtype=float)

    B, S, d_in = X.shape
    d_out = W_q.shape[1]

    if d_out % num_heads != 0:
        raise ValueError("d_out must be divisible by num_heads")

    d_head = d_out // num_heads

    # 1. Q, K, V projections with einsum.
    Q = np.einsum("bsd,df->bsf", X, W_q)
    K = np.einsum("bsd,df->bsf", X, W_k)
    V = np.einsum("bsd,df->bsf", X, W_v)

    # 2. Split into heads: (B, S, d_out) -> (B, H, S, d_head)
    Q = Q.reshape(B, S, num_heads, d_head).transpose(0, 2, 1, 3)
    K = K.reshape(B, S, num_heads, d_head).transpose(0, 2, 1, 3)
    V = V.reshape(B, S, num_heads, d_head).transpose(0, 2, 1, 3)

    # 3. Scaled attention scores.
    scores = np.einsum("bhid,bhjd->bhij", Q, K) / np.sqrt(d_head)

    # 4. Causal mask.
    if causal:
        mask = np.triu(
            np.ones((S, S), dtype=bool),
            k=1
        )
        scores = np.where(mask[None, None, :, :], -np.inf, scores)

    # 5. Stable softmax over the key dimension.
    max_scores = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    attn = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # 6. Aggregate values.
    context = np.einsum("bhij,bhjd->bhid", attn, V)

    # Merge heads: (B, H, S, d_head) -> (B, S, d_out)
    context = context.transpose(0, 2, 1, 3).reshape(B, S, d_out)

    # 7. Output projection with einsum.
    output = np.einsum("bsd,df->bsf", context, W_o)

    return output.tolist()