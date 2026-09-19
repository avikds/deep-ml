import numpy as np

def train_simple_cnn_with_backprop(
    X, y, epochs, learning_rate, kernel_size=3, num_filters=1
):
    '''
    Trains a simple CNN with one convolutional layer, ReLU activation,
    flattening, and a dense layer with softmax output using SGD.

    X: shape (n_samples, height, width)
    y: one-hot encoded, shape (n_samples, num_classes)
    '''

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    n_samples, height, width = X.shape
    num_classes = y.shape[1]

    # Initialize weights and biases
    W_conv = np.random.randn(
        kernel_size, kernel_size, num_filters
    ) * 0.01
    b_conv = np.zeros(num_filters)

    output_height = height - kernel_size + 1
    output_width = width - kernel_size + 1
    flattened_size = output_height * output_width * num_filters

    W_dense = np.random.randn(
        flattened_size, num_classes
    ) * 0.01
    b_dense = np.zeros(num_classes)

    for _ in range(epochs):
        for i in range(n_samples):
            image = X[i]
            target = y[i]

            # ---------------------------------------------------------
            # Forward pass
            # ---------------------------------------------------------
            conv = np.zeros(
                (output_height, output_width, num_filters)
            )

            for f in range(num_filters):
                kernel = W_conv[:, :, f]

                for r in range(output_height):
                    for c in range(output_width):
                        region = image[
                            r:r + kernel_size,
                            c:c + kernel_size
                        ]
                        conv[r, c, f] = (
                            np.sum(region * kernel) + b_conv[f]
                        )

            # ReLU
            relu = np.maximum(0.0, conv)

            # Flatten
            flat = relu.reshape(-1)

            # Dense
            logits = flat @ W_dense + b_dense

            # Stable softmax
            shifted = logits - np.max(logits)
            exp_logits = np.exp(shifted)
            probs = exp_logits / np.sum(exp_logits)

            # ---------------------------------------------------------
            # Backward pass
            # ---------------------------------------------------------
            # Cross-entropy gradient wrt logits
            d_logits = probs - target

            # Dense gradients
            dW_dense = np.outer(flat, d_logits)
            db_dense = d_logits

            # Gradient wrt flattened ReLU output
            d_flat = W_dense @ d_logits
            d_relu = d_flat.reshape(
                output_height, output_width, num_filters
            )

            # ReLU derivative
            d_conv = d_relu * (conv > 0)

            # Convolution gradients
            dW_conv = np.zeros_like(W_conv)
            db_conv = np.zeros_like(b_conv)

            for f in range(num_filters):
                for r in range(output_height):
                    for c in range(output_width):
                        grad = d_conv[r, c, f]

                        region = image[
                            r:r + kernel_size,
                            c:c + kernel_size
                        ]

                        dW_conv[:, :, f] += grad * region
                        db_conv[f] += grad

            # ---------------------------------------------------------
            # SGD update
            # ---------------------------------------------------------
            W_conv -= learning_rate * dW_conv
            b_conv -= learning_rate * db_conv
            W_dense -= learning_rate * dW_dense
            b_dense -= learning_rate * db_dense

    return W_conv, b_conv, W_dense, b_dense