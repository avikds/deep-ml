import numpy as np

def embedding_via_one_hot(token_ids, W):
    """
    Compute token embeddings via one-hot encoding and matrix multiplication.
    """
    token_ids = np.asarray(token_ids, dtype=int)
    vocab_size = W.shape[0]

    H = np.zeros((len(token_ids), vocab_size), dtype=float)
    H[np.arange(len(token_ids)), token_ids] = 1.0

    return H @ W