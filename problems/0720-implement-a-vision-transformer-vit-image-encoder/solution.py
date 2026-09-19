import numpy as np

def vit_encode(image, patch_size: int, W_patch, cls_token, pos_embed, W_qkv, W_o, num_heads: int):
    """
    Forward pass of a minimal ViT encoder block.

    Returns the final embedding of the CLS token as a list of length D.
    """
    image = np.asarray(image, dtype=float)
    W_patch = np.asarray(W_patch, dtype=float)
    cls_token = np.asarray(cls_token, dtype=float)
    pos_embed = np.asarray(pos_embed, dtype=float)
    W_qkv = np.asarray(W_qkv, dtype=float)
    W_o = np.asarray(W_o, dtype=float)

    H, W, C = image.shape
    D = cls_token.shape[0]

    # Create flattened patch embeddings in row-major (C-order).
    patches = []
    for r in range(0, H, patch_size):
        for c in range(0, W, patch_size):
            patch = image[r:r + patch_size, c:c + patch_size, :]
            patches.append(patch.reshape(-1))

    patches = np.asarray(patches, dtype=float)
    patch_embeddings = patches @ W_patch

    # Prepend CLS token and add positional embeddings.
    tokens = np.vstack([cls_token, patch_embeddings])
    tokens = tokens + pos_embed

    # Q, K, V projections.
    qkv = tokens @ W_qkv
    Q = qkv[:, :D]
    K = qkv[:, D:2 * D]
    V = qkv[:, 2 * D:]

    # Multi-head self-attention.
    head_dim = D // num_heads

    # (sequence, heads, head_dim) -> (heads, sequence, head_dim)
    Q = Q.reshape(-1, num_heads, head_dim).transpose(1, 0, 2)
    K = K.reshape(-1, num_heads, head_dim).transpose(1, 0, 2)
    V = V.reshape(-1, num_heads, head_dim).transpose(1, 0, 2)

    # Scaled dot-product attention.
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(head_dim)

    # Numerically stable softmax.
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    attn = np.exp(scores)
    attn /= np.sum(attn, axis=-1, keepdims=True)

    context = np.matmul(attn, V)

    # (heads, sequence, head_dim) -> (sequence, D)
    context = context.transpose(1, 0, 2).reshape(-1, D)

    # Output projection.
    output = context @ W_o

    # Return CLS token embedding.
    return output[0].tolist()