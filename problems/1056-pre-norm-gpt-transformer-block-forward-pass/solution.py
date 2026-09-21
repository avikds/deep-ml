import numpy as np

def pre_norm_transformer_block(x, params, num_heads):
    """
    Pre-norm Transformer block forward pass.
    """
    x = np.asarray(x, dtype=float).copy()

    ln1_gamma = np.asarray(params["ln1_gamma"], dtype=float)
    ln1_beta = np.asarray(params["ln1_beta"], dtype=float)
    ln2_gamma = np.asarray(params["ln2_gamma"], dtype=float)
    ln2_beta = np.asarray(params["ln2_beta"], dtype=float)

    W_q = np.asarray(params["W_q"], dtype=float)
    W_k = np.asarray(params["W_k"], dtype=float)
    W_v = np.asarray(params["W_v"], dtype=float)
    W_o = np.asarray(params["W_o"], dtype=float)

    W_ff1 = np.asarray(params["W_ff1"], dtype=float)
    b_ff1 = np.asarray(params["b_ff1"], dtype=float)
    W_ff2 = np.asarray(params["W_ff2"], dtype=float)
    b_ff2 = np.asarray(params["b_ff2"], dtype=float)

    B, T, D = x.shape

    if D % num_heads != 0:
        raise ValueError("emb_dim must be divisible by num_heads")

    d_head = D // num_heads
    eps = 1e-5

    def layer_norm(z, gamma, beta):
        mean = np.mean(z, axis=-1, keepdims=True)
        var = np.mean((z - mean) ** 2, axis=-1, keepdims=True)
        return ((z - mean) / np.sqrt(var + eps)) * gamma + beta

    # ---------------------------------------------------------
    # 1. Attention sub-block: x <- x + MHA(LN1(x))
    # ---------------------------------------------------------
    h = layer_norm(x, ln1_gamma, ln1_beta)

    Q = h @ W_q
    K = h @ W_k
    V = h @ W_v

    # (B, T, D) -> (B, H, T, d_head)
    Q = Q.reshape(B, T, num_heads, d_head).transpose(0, 2, 1, 3)
    K = K.reshape(B, T, num_heads, d_head).transpose(0, 2, 1, 3)
    V = V.reshape(B, T, num_heads, d_head).transpose(0, 2, 1, 3)

    scores = np.matmul(Q, np.swapaxes(K, -1, -2)) / np.sqrt(d_head)

    # Causal mask: future positions cannot be attended to.
    causal_mask = np.triu(
        np.ones((T, T), dtype=bool),
        k=1
    )
    scores = np.where(
        causal_mask[None, None, :, :],
        -np.inf,
        scores
    )

    # Stable softmax.
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attn = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    context = np.matmul(attn, V)

    # (B, H, T, d_head) -> (B, T, D)
    context = context.transpose(0, 2, 1, 3).reshape(B, T, D)

    h = context @ W_o

    # Running residual stream.
    x = x + h

    # ---------------------------------------------------------
    # 2. FFN sub-block: x <- x + FFN(LN2(x))
    # ---------------------------------------------------------
    h = layer_norm(x, ln2_gamma, ln2_beta)

    z = h @ W_ff1 + b_ff1

    # Tanh GELU approximation.
    gelu = 0.5 * z * (
        1.0
        + np.tanh(
            np.sqrt(2.0 / np.pi)
            * (z + 0.044715 * z**3)
        )
    )

    h = gelu @ W_ff2 + b_ff2

    x = x + h

    return x