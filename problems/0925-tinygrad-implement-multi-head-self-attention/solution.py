from tinygrad import Tensor, nn

class MultiHeadSelfAttention:
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0

        self.d_head = d_model // num_heads
        self.num_heads = num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def __call__(self, x, mask=None):
        B, T, _ = x.shape

        # Project to Q, K, V.
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # (B, T, D) -> (B, H, T, d_head)
        q = q.reshape(B, T, self.num_heads, self.d_head).transpose(1, 2)
        k = k.reshape(B, T, self.num_heads, self.d_head).transpose(1, 2)
        v = v.reshape(B, T, self.num_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product attention.
        scores = (q @ k.transpose(2, 3)) / self.d_head

        # Add optional attention mask.
        if mask is not None:
            scores = scores + mask

        # Softmax over keys.
        attn = scores.softmax(axis=-1)

        # Weighted values.
        out = attn @ v

        # (B, H, T, d_head) -> (B, T, D)
        out = out.transpose(1, 2).reshape(B, T, self.num_heads * self.d_head)

        return self.out_proj(out)