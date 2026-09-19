import numpy as np

def spatiotemporal_transformer_block(X: np.ndarray, params: dict) -> np.ndarray:
    """
    Forward pass of a spatiotemporal transformer block.
    """
    X = np.asarray(X, dtype=float)

    eps = 1e-5

    def layer_norm(x):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + eps)

    def self_attention(x, Wq, Wk, Wv, Wo):
        Q = x @ Wq
        K = x @ Wk
        V = x @ Wv

        D = x.shape[-1]

        # Attention over the second-to-last dimension.
        scores = np.matmul(Q, np.swapaxes(K, -1, -2)) / np.sqrt(D)

        # Stable softmax.
        scores = scores - np.max(scores, axis=-1, keepdims=True)
        weights = np.exp(scores)
        weights /= np.sum(weights, axis=-1, keepdims=True)

        context = weights @ V
        return context @ Wo

    # 1. Spatial self-attention:
    # independently attend over N tokens within each frame.
    x_norm = layer_norm(X)
    spatial_out = self_attention(
        x_norm,
        np.asarray(params["Wqs"], dtype=float),
        np.asarray(params["Wks"], dtype=float),
        np.asarray(params["Wvs"], dtype=float),
        np.asarray(params["Wos"], dtype=float),
    )
    X = X + spatial_out

    # 2. Temporal self-attention:
    # independently attend over T frames for each spatial position.
    x_norm = layer_norm(X)

    # (T, N, D) -> (N, T, D)
    temporal_input = np.transpose(x_norm, (1, 0, 2))

    temporal_out = self_attention(
        temporal_input,
        np.asarray(params["Wqt"], dtype=float),
        np.asarray(params["Wkt"], dtype=float),
        np.asarray(params["Wvt"], dtype=float),
        np.asarray(params["Wot"], dtype=float),
    )

    # Back to (T, N, D).
    temporal_out = np.transpose(temporal_out, (1, 0, 2))
    X = X + temporal_out

    # 3. Position-wise FFN with ReLU.
    x_norm = layer_norm(X)

    W1 = np.asarray(params["W1"], dtype=float)
    W2 = np.asarray(params["W2"], dtype=float)

    hidden = np.maximum(x_norm @ W1, 0.0)
    ffn_out = hidden @ W2

    X = X + ffn_out

    return X