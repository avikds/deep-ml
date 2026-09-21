import numpy as np

def cross_layer_kv_attention(x, layer_weights, n_kv_producing_layers):
    """
    Multi-layer causal self-attention with cross-layer KV sharing.
    """
    h = np.asarray(x, dtype=float).copy()

    seq_len, d_model = h.shape
    n_layers = len(layer_weights)

    shared_k = None
    shared_v = None

    # Causal mask: positions may only attend to themselves and earlier
    # positions. Future positions receive a large negative score.
    causal_mask = np.triu(
        np.full((seq_len, seq_len), -1e9, dtype=float),
        k=1
    )

    for i in range(n_layers):
        Wq = np.asarray(layer_weights[i]["Wq"], dtype=float)

        # Each layer always computes its own Q from the current hidden state.
        Q = h @ Wq

        if i < n_kv_producing_layers:
            Wk = np.asarray(layer_weights[i]["Wk"], dtype=float)
            Wv = np.asarray(layer_weights[i]["Wv"], dtype=float)

            # Fresh K/V from the current layer's hidden state.
            shared_k = h @ Wk
            shared_v = h @ Wv

        # Causal scaled dot-product attention.
        scores = (Q @ shared_k.T) / np.sqrt(d_model)
        scores = scores + causal_mask

        # Numerically stable softmax.
        scores = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        attn_weights = exp_scores / np.sum(
            exp_scores, axis=1, keepdims=True
        )

        attn = attn_weights @ shared_v

        # Residual connection.
        h = h + attn

    return np.round(h, 4).tolist()