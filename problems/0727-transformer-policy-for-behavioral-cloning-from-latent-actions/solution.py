import numpy as np

def bc_transformer_loss(obs: np.ndarray, targets: np.ndarray, weights: dict) -> float:
    """
    Compute the behavioral cloning loss of a single-block transformer policy
    predicting discrete latent actions from a sequence of observation embeddings.
    """
    obs = np.asarray(obs, dtype=float)
    targets = np.asarray(targets, dtype=int)

    W_q = np.asarray(weights["W_q"], dtype=float)
    W_k = np.asarray(weights["W_k"], dtype=float)
    W_v = np.asarray(weights["W_v"], dtype=float)
    W_o = np.asarray(weights["W_o"], dtype=float)
    W1 = np.asarray(weights["W1"], dtype=float)
    b1 = np.asarray(weights["b1"], dtype=float)
    W2 = np.asarray(weights["W2"], dtype=float)
    b2 = np.asarray(weights["b2"], dtype=float)
    W_head = np.asarray(weights["W_head"], dtype=float)
    b_head = np.asarray(weights["b_head"], dtype=float)

    T, d_model = obs.shape

    # 1. Causal self-attention.
    Q = obs @ W_q
    K = obs @ W_k
    V = obs @ W_v

    scores = (Q @ K.T) / np.sqrt(d_model)

    # Mask future positions with -inf.
    causal_mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    scores = np.where(causal_mask, -np.inf, scores)

    # Stable softmax row-wise.
    max_scores = np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    attn = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    attn_out = attn @ V
    attn_out = attn_out @ W_o

    # First residual.
    x = obs + attn_out

    # 2. Position-wise MLP with second residual.
    hidden = np.maximum(x @ W1 + b1, 0.0)
    mlp_out = hidden @ W2 + b2

    x = x + mlp_out

    # 3. Policy head.
    logits = x @ W_head + b_head

    # Mean token-level cross-entropy.
    logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
    logsumexp = np.log(np.sum(np.exp(logits_shifted), axis=1))
    log_probs = logits_shifted - logsumexp[:, None]

    loss = -np.mean(log_probs[np.arange(T), targets])

    return float(loss)