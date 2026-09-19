import torch
import torch.nn as nn

def generate_adversarial_example(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    criterion: nn.Module
) -> torch.Tensor:
    '''
    Generate an adversarial example using iterative PGD.

    Args:
        model: Pre-trained classifier.
        x: Input image tensor, shape (1, 1, 28, 28), values in [0, 1].
        y: True label, shape (1,) or scalar.
        epsilon: L-infinity perturbation budget.
        criterion: Loss function, e.g. nn.CrossEntropyLoss().

    Returns:
        x_adv: Same shape as x, with:
               ||x_adv - x||_inf <= epsilon
               x_adv in [0, 1]
    '''

    # Preserve the original input.
    x_orig = x.detach()

    # Make sure y has batch dimension.
    if y.dim() == 0:
        y = y.unsqueeze(0)

    # Step size for PGD.
    # Multiple smaller steps are more reliable than one large step.
    num_steps = 20
    alpha = epsilon / 5.0

    # Start from the original image.
    x_adv = x_orig.clone()

    # Random start inside the epsilon-ball.
    if epsilon > 0:
        noise = torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
        x_adv = torch.clamp(x_adv + noise, 0.0, 1.0)

    for _ in range(num_steps):
        x_adv.requires_grad_(True)

        # Forward pass
        output = model(x_adv)

        # Stop immediately if the current example already fools the model.
        with torch.no_grad():
            if output.argmax(dim=1).item() != y.item():
                return x_adv.detach()

        # Compute gradient of classification loss with respect to input.
        loss = criterion(output, y)

        model.zero_grad(set_to_none=True)

        if x_adv.grad is not None:
            x_adv.grad.zero_()

        loss.backward()

        # Gradient ascent: maximize classification loss.
        grad = x_adv.grad.detach()

        # FGSM/PGD sign update.
        x_adv = x_adv.detach() + alpha * grad.sign()

        # Project back into the L-infinity epsilon-ball
        # around the original image.
        delta = torch.clamp(
            x_adv - x_orig,
            min=-epsilon,
            max=epsilon
        )

        x_adv = x_orig + delta

        # Keep pixels in the valid image range.
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    return x_adv.detach()