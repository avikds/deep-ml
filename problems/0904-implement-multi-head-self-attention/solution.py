import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_head = d_model // num_heads
        self.num_heads = num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        # x: (B, T, d_model)
        B, T, _ = x.shape

        # Project to Q, K, V.
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # (B, T, d_model) -> (B, num_heads, T, d_head)
        q = q.view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product attention.
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.d_head

        if mask is not None:
            scores = scores + mask

        attn = F.softmax(scores, dim=-1)

        # Weighted sum of values.
        out = torch.matmul(attn, v)

        # (B, num_heads, T, d_head) -> (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(
            B, T, self.num_heads * self.d_head
        )

        return self.out_proj(out)