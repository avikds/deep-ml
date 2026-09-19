from tinygrad import Tensor, nn

class TinyTransformerBlock:
    def __init__(self, d_model, num_heads, d_ff):
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # Pre-LN attention
        self.ln1 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Pre-LN MLP
        self.ln2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def __call__(self, x):
        B, T, _ = x.shape

        # Pre-LN causal self-attention.
        h = self.ln1(x)

        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)

        q = q.reshape(B, T, self.num_heads, self.d_head).transpose(1, 2)
        k = k.reshape(B, T, self.num_heads, self.d_head).transpose(1, 2)
        v = v.reshape(B, T, self.num_heads, self.d_head).transpose(1, 2)

        attn = q.scaled_dot_product_attention(
            k, v, is_causal=True
        )

        attn = attn.transpose(1, 2).reshape(
            B, T, self.d_model
        )

        x = x + self.out_proj(attn)

        # Pre-LN feed-forward network.
        h = self.ln2(x)
        h = self.fc1(h).relu()
        h = self.fc2(h)

        x = x + h

        return x


class TinyGPT:
    def __init__(self, vocab_size, d_model, num_heads, num_layers, max_seq_len):
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        self.blocks = [
            TinyTransformerBlock(
                d_model,
                num_heads,
                4 * d_model
            )
            for _ in range(num_layers)
        ]

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def __call__(self, idx):
        B, T = idx.shape

        # Token embeddings + learned positional embeddings.
        positions = Tensor.arange(T)
        x = self.token_emb(idx) + self.pos_emb(positions)

        # Transformer stack.
        for block in self.blocks:
            x = block(x)

        # Final normalization and vocabulary projection.
        x = self.ln_f(x)
        return self.head(x)