import numpy as np

def attention(Q, K, V):
    """
    Compute scaled dot-product attention.
    """
    Q = np.asarray(Q, dtype=float)
    K = np.asarray(K, dtype=float)
    V = np.asarray(V, dtype=float)

    _, _, dim = Q.shape

    # Query-key compatibility scores
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(dim)

    # Numerically stable softmax over key positions
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores)
    weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # Weighted combination of values
    output = np.matmul(weights, V)

    return output