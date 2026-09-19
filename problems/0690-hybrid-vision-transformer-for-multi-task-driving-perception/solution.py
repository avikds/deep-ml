import numpy as np

def hybrid_vit_multitask(image, patch_conv_w, patch_conv_b, pos_emb,
                         attn_Wq, attn_Wk, attn_Wv, attn_Wo,
                         num_heads, task_heads):
    """
    Hybrid Vision Transformer for multi-task driving perception.
    """
    image = np.asarray(image, dtype=float)
    patch_conv_w = np.asarray(patch_conv_w, dtype=float)
    patch_conv_b = np.asarray(patch_conv_b, dtype=float)
    pos_emb = np.asarray(pos_emb, dtype=float)
    attn_Wq = np.asarray(attn_Wq, dtype=float)
    attn_Wk = np.asarray(attn_Wk, dtype=float)
    attn_Wv = np.asarray(attn_Wv, dtype=float)
    attn_Wo = np.asarray(attn_Wo, dtype=float)

    H, W, C = image.shape
    D, kH, kW, _ = patch_conv_w.shape

    # 1. Convolutional patch embedding.
    # Stride == kernel size => non-overlapping patches.
    tokens = []

    for r in range(0, H, kH):
        for c in range(0, W, kW):
            patch = image[r:r + kH, c:c + kW, :]

            # Each output channel/filter produces one scalar.
            # Equivalent to a valid convolution at this patch location.
            embedding = np.sum(
                patch[None, :, :, :] * patch_conv_w,
                axis=(1, 2, 3)
            ) + patch_conv_b

            tokens.append(embedding)

    tokens = np.asarray(tokens, dtype=float)

    # 2. Positional embedding.
    tokens = tokens + pos_emb

    # Save pre-attention tokens for residual connection.
    residual = tokens.copy()

    # 3. Multi-head self-attention.
    Q = tokens @ attn_Wq
    K = tokens @ attn_Wk
    V = tokens @ attn_Wv

    n_tokens = tokens.shape[0]
    head_dim = D // num_heads

    Q = Q.reshape(n_tokens, num_heads, head_dim).transpose(1, 0, 2)
    K = K.reshape(n_tokens, num_heads, head_dim).transpose(1, 0, 2)
    V = V.reshape(n_tokens, num_heads, head_dim).transpose(1, 0, 2)

    # Scaled dot-product attention.
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(head_dim)

    # Numerically stable row-wise softmax.
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores)
    attention = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    context = np.matmul(attention, V)

    # Concatenate heads.
    context = context.transpose(1, 0, 2).reshape(n_tokens, D)

    # Output projection + residual.
    features = residual + context @ attn_Wo

    # 4. Task-specific prediction heads.
    outputs = {}

    for name, (W, b) in task_heads.items():
        W = np.asarray(W, dtype=float)
        b = np.asarray(b, dtype=float)
        outputs[name] = features @ W + b

    return outputs