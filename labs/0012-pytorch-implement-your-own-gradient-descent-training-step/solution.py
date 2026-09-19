import torch
import torch.nn as nn
import torch.nn.functional as F

def train_step(model, x_batch, y_batch, lr):
    """
    Perform ONE step of manual gradient descent.
    """

    # Step 1: Zero existing gradients
    for param in model.parameters():
        if param.grad is not None:
            param.grad.zero_()

    # Step 2: Forward pass
    logits = model(x_batch)

    # Step 3: Compute cross-entropy loss
    loss = F.cross_entropy(logits, y_batch)

    # Step 4: Backpropagation
    loss.backward()

    # Step 5: Gradient descent parameter update
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                param -= lr * param.grad

    # Step 6: Return Python float
    return loss.item()