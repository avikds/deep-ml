import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, lr):
    history = []

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    n = X_train.shape[0]

    for epoch in range(1, epochs + 1):
        model.train()

        # Shuffle training data each epoch
        indices = torch.randperm(n)

        total_loss = 0.0
        total_samples = 0

        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]

            xb = X_train[batch_idx]
            yb = y_train[batch_idx]

            optimizer.zero_grad()

            logits = model(xb)
            loss = criterion(logits, yb)

            loss.backward()
            optimizer.step()

            bs = xb.shape[0]
            total_loss += loss.item() * bs
            total_samples += bs

        train_loss = total_loss / total_samples

        # Validation
        model.eval()

        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = criterion(val_logits, y_val).item()
            predictions = val_logits.argmax(dim=1)
            val_accuracy = (predictions == y_val).float().mean().item()

        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_accuracy)
        })

    return history