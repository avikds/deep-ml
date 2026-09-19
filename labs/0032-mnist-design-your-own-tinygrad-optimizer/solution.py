from tinygrad import Tensor
from tinygrad.nn.optim import Optimizer


class MyOptimizer(Optimizer):
    """
    Adam-style optimizer with decoupled weight decay.
    Designed to be stable for small CNNs on MNIST.
    """

    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999,
                 eps=1e-8, weight_decay=1e-4):
        super().__init__(params, lr)

        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        # First- and second-moment buffers
        self.m = [Tensor.zeros_like(p, requires_grad=False) for p in self.params]
        self.v = [Tensor.zeros_like(p, requires_grad=False) for p in self.params]

    def schedule_step(self):
        self.t += 1

        b1 = self.beta1
        b2 = self.beta2
        lr = self.lr
        eps = self.eps
        wd = self.weight_decay

        realized = []

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            g = p.grad.detach()

            # Update exponential moving averages
            self.m[i].assign(b1 * self.m[i] + (1.0 - b1) * g)
            self.v[i].assign(b2 * self.v[i] + (1.0 - b2) * g * g)

            # Bias correction
            m_hat = self.m[i] / (1.0 - b1 ** self.t)
            v_hat = self.v[i] / (1.0 - b2 ** self.t)

            # Adam update + decoupled weight decay
            new_p = (
                p.detach() * (1.0 - lr * wd)
                - lr * m_hat / (v_hat.sqrt() + eps)
            )

            p.assign(new_p)

            realized.append(p)
            realized.append(self.m[i])
            realized.append(self.v[i])

        return realized