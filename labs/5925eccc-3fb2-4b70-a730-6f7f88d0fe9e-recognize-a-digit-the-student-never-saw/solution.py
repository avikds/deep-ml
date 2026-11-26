import numpy as np
import torch
import torch.nn.functional as F
import math


def train(X, y, teacher_logits):
    """Distill teacher knowledge and calibrate the missing class."""

    X = torch.as_tensor(X, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.long)
    teacher = torch.as_tensor(
        teacher_logits, dtype=torch.float32
    )

    n = len(X)

    model = torch.nn.Sequential(
        torch.nn.Linear(784, 400),
        torch.nn.ReLU(),
        torch.nn.Linear(400, 400),
        torch.nn.ReLU(),
        torch.nn.Linear(400, 10)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0012,
        weight_decay=0.00001
    )

    teacher = teacher - teacher.mean(dim=1, keepdim=True)

    temperature = 3.0

    with torch.no_grad():
        teacher_probs = F.softmax(
            teacher / temperature, dim=1
        )
        teacher_normal = F.softmax(teacher, dim=1)

    epochs = 80
    batch_size = 512

    for epoch in range(epochs):
        progress = epoch / epochs

        lr = 0.00008 + 0.00112 * (
            1.0 + math.cos(math.pi * progress)
        ) / 2.0

        for group in optimizer.param_groups:
            group["lr"] = lr

        model.train()

        permutation = torch.randperm(n)

        for start in range(0, n, batch_size):
            idx = permutation[start:start + batch_size]

            xb = X[idx]
            yb = y[idx]
            tb = teacher[idx]
            tp = teacher_probs[idx]
            tn = teacher_normal[idx]

            logits = model(xb)

            loss_mse = F.mse_loss(logits, tb)

            loss_soft = F.kl_div(
                F.log_softmax(logits / temperature, dim=1),
                tp,
                reduction="batchmean"
            ) * temperature ** 2

            loss_normal = F.kl_div(
                F.log_softmax(logits, dim=1),
                tn,
                reduction="batchmean"
            )

            loss_hard = F.cross_entropy(logits, yb)

            loss = (
                0.75 * loss_mse
                + 0.20 * loss_soft
                + 0.04 * loss_normal
                + 0.01 * loss_hard
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5.0
            )

            optimizer.step()

    # Extract the trained network.
    with torch.no_grad():
        model.eval()
        student_logits = model(X)

        W1 = model[0].weight.numpy().copy()
        b1 = model[0].bias.numpy().copy()

        W2 = model[2].weight.numpy().copy()
        b2 = model[2].bias.numpy().copy()

        W3 = model[4].weight.numpy().copy()
        b3 = model[4].bias.numpy().copy()

        student_logits = student_logits.numpy()

        teacher_np = teacher.numpy()

    # Estimate teacher's false-positive rate for class 3.
    teacher_margin = (
        teacher_np[:, 3]
        - np.max(
            np.concatenate(
                [teacher_np[:, :3], teacher_np[:, 4:]],
                axis=1
            ),
            axis=1
        )
    )

    target_rate = np.mean(teacher_margin > 0)

    # Student's class-3 decision margins.
    student_margin = (
        student_logits[:, 3]
        - np.max(
            np.concatenate(
                [student_logits[:, :3], student_logits[:, 4:]],
                axis=1
            ),
            axis=1
        )
    )

    # Match the teacher's false-positive rate.
    target_rate = np.clip(
        target_rate, 1.0 / n, 1.0 - 1.0 / n
    )

    threshold = np.quantile(
        student_margin,
        1.0 - target_rate
    )

    shift3 = -float(threshold)

    return {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2,
        "W3": W3,
        "b3": b3,
        "shift3": shift3
    }


def predict(model, X):
    """Predict ten MNIST classes with class-3 calibration."""

    X = np.asarray(X, dtype=np.float32)

    h1 = np.maximum(
        X @ model["W1"].T + model["b1"], 0.0
    )

    h2 = np.maximum(
        h1 @ model["W2"].T + model["b2"], 0.0
    )

    scores = h2 @ model["W3"].T + model["b3"]

    # Correct the missing-class decision threshold.
    scores[:, 3] += model["shift3"]

    return scores