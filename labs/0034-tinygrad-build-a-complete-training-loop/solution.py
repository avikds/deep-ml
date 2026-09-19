from tinygrad import Tensor
from tinygrad.nn.state import get_parameters
from tinygrad.nn.optim import Adam


def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, lr):
    """
    Train a tinygrad model and return per-epoch metrics.
    """
    history = []

    # Create optimizer
    params = get_parameters(model)
    optimizer = Adam(params, lr=lr)

    n_train = X_train.shape[0]

    for epoch in range(epochs):
        # Shuffle using tinygrad Tensor indexing
        perm = Tensor.randperm(n_train)

        # Training mode
        Tensor.training = True

        total_train_loss = 0.0
        total_train_samples = 0

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)

            batch_idx = perm[start:end]
            X_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]

            # Reset gradients
            optimizer.zero_grad()

            # Forward
            logits = model(X_batch)

            # Required loss
            loss = logits.sparse_categorical_crossentropy(y_batch)

            # Backward
            loss.backward()

            # Some models can contain parameters that did not participate
            # in this particular computation. tinygrad's optimizer expects
            # every parameter to have a gradient.
            for p in params:
                if p.grad is None:
                    p.grad = Tensor.zeros_like(p)

            # Update
            optimizer.step()

            # Record training loss
            batch_n = end - start
            total_train_loss += loss.item() * batch_n
            total_train_samples += batch_n

        train_loss = total_train_loss / total_train_samples

        # Validation mode
        Tensor.training = False

        val_logits = model(X_val)

        val_loss = val_logits.sparse_categorical_crossentropy(y_val).item()

        predictions = val_logits.argmax(axis=1)
        val_accuracy = (predictions == y_val).mean().item()

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        })

    return history