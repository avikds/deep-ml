import torch
import torch.nn.functional as F
import numpy as np

def init_aux_params(d_model: int, vocab_size: int, seed: int = 0) -> dict:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    scale = 0.02

    W_h = (torch.randn(d_model, d_model, generator=g) * scale).requires_grad_()
    W_e = (torch.randn(d_model, d_model, generator=g) * scale).requires_grad_()
    b = torch.zeros(d_model, requires_grad=True)

    return {
        "W_h": W_h,
        "W_e": W_e,
        "b": b,
    }


def latent_objective(h, x, mask, params, aux):
    # Predict h[t+1] from h[t] and the token consumed at t.
    B, T, d = h.shape

    if T < 2:
        return h.new_zeros(())

    h_prev = h[:, :-1, :]          # (B, T-1, d)
    h_next = h[:, 1:, :]           # (B, T-1, d)

    # Token at position t is what has just been consumed to move from
    # h[t] to h[t+1].
    x_cur = x[:, :-1]              # (B, T-1)
    x_emb = params["wte"].detach()[x_cur]  # (B, T-1, d)

    pred = (
        h_prev
        + torch.matmul(h_prev, aux["W_h"].T)
        + torch.matmul(x_emb, aux["W_e"].T)
        + aux["b"]
    )

    err = ((pred - h_next) ** 2).mean(dim=-1)  # (B, T-1)

    valid = mask[:, :-1] & mask[:, 1:]        # (B, T-1)

    if torch.any(valid):
        return 0.01 * err[valid].mean()

    return h.new_zeros(())


def latent_transition(h, x_emb, aux):
    return (
        h
        + torch.matmul(h, aux["W_h"].T)
        + torch.matmul(x_emb, aux["W_e"].T)
        + aux["b"]
    )