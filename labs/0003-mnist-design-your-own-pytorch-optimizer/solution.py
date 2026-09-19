import math
import torch
from torch.optim.optimizer import Optimizer


class MyOptimizer(Optimizer):
    """
    Adam-style optimizer with decoupled weight decay.
    Stable for training a small CNN on MNIST.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=1e-4,
    ):
        defaults = dict(
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                if p.grad.is_sparse:
                    raise RuntimeError("MyOptimizer does not support sparse gradients")

                grad = p.grad

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                state["step"] += 1
                step = state["step"]

                # Update first and second moments
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(
                    grad, grad, value=1.0 - beta2
                )

                # Bias correction
                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step

                step_size = lr / bias_correction1
                denom = exp_avg_sq.sqrt() / math.sqrt(bias_correction2)
                denom.add_(eps)

                # Decoupled weight decay
                if weight_decay != 0:
                    p.mul_(1.0 - lr * weight_decay)

                # Parameter update
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss