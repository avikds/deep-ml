from tinygrad import Tensor
from tinygrad.nn.state import get_parameters

def train_step(model, x_batch, y_batch, lr):
    params = get_parameters(model)

    for p in params:
        p.requires_grad = True
        p.grad = None

    logits = model(x_batch)
    loss = logits.sparse_categorical_crossentropy(y_batch)

    loss.backward()

    loss_value = float(loss.item())

    for p in params:
        if p.grad is not None:
            p.assign(p.detach() - lr * p.grad.detach())

    Tensor.realize(*params)

    return loss_value