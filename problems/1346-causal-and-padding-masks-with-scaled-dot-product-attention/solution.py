import torch
import torch.nn as nn
import torch.nn.functional as F


def attention(q, k, v, key_padding_mask=None, causal=False):
    B, H, Tq, D = q.shape
    Tk = k.shape[2]

    attn_mask = None

    if key_padding_mask is not None:
        # True means valid key; reshape for broadcasting.
        attn_mask = key_padding_mask[:, None, None, :]

    if causal:
        # Lower-triangular causal mask.
        causal_mask = torch.ones(
            (Tq, Tk),
            dtype=torch.bool,
            device=q.device
        ).tril()

        causal_mask = causal_mask[None, None, :, :]

        if attn_mask is None:
            attn_mask = causal_mask
        else:
            attn_mask = attn_mask & causal_mask

    return F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attn_mask
    )