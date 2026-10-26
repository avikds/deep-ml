import torch
import torch.nn.functional as F


def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Numerically robust cross-entropy using target-relative logits.
    Keeps calculations in the input dtype (float32 in this task).
    """
    target_logits = logits.gather(1, targets.reshape(-1, 1))
    shifted = logits - target_logits

    # Exclude the target class from the competing exponentials.
    shifted = shifted.scatter(
        1,
        targets.reshape(-1, 1),
        float("-inf"),
    )

    # Equivalent to log(1 + sum_{j != y} exp(logit_j - logit_y)).
    zeros = torch.zeros_like(target_logits)
    per_example_loss = torch.logsumexp(
        torch.cat((zeros, shifted), dim=1),
        dim=1,
    )

    return per_example_loss.mean()


def make_optimizer(
    params,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """AdamW with the provided learning rate and weight decay."""
    return torch.optim.AdamW(
        list(params),
        lr=lr,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=weight_decay,
    )