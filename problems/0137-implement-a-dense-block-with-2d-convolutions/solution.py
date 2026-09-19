import numpy as np

def dense_net_block(input_data, num_layers, growth_rate, kernels, kernel_size=(3, 3)):
    x = np.asarray(input_data)

    kh, kw = kernel_size

    # Validate number of kernels
    if len(kernels) < num_layers:
        raise ValueError("Not enough kernels for num_layers")

    for l in range(num_layers):
        kernel = np.asarray(kernels[l])

        # Check kernel spatial dimensions
        if kernel.ndim != 4 or kernel.shape[0] != kh or kernel.shape[1] != kw:
            raise ValueError("Kernel shape does not match kernel_size")

        # Current channel count
        current_channels = x.shape[-1]

        # Check input-channel dimension
        if kernel.shape[2] != current_channels:
            raise ValueError(
                f"Kernel {l} expects {kernel.shape[2]} input channels, "
                f"but current feature map has {current_channels}"
            )

        # Check output-channel dimension
        if kernel.shape[3] != growth_rate:
            raise ValueError(
                f"Kernel {l} must have {growth_rate} output channels"
            )

        # ReLU
        relu_x = np.maximum(x, 0)

        # Symmetric zero padding to preserve H and W.
        pad_h = kh // 2
        pad_w = kw // 2

        padded = np.pad(
            relu_x,
            ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
            mode="constant"
        )

        n, h, w, _ = x.shape
        out = np.zeros((n, h, w, growth_rate), dtype=np.result_type(x, kernel))

        # NHWC convolution, stride 1, no bias
        for i in range(h):
            for j in range(w):
                region = padded[:, i:i + kh, j:j + kw, :]
                out[:, i, j, :] = np.einsum(
                    "nijk,ijkc->nc",
                    region,
                    kernel
                )

        # DenseNet concatenation
        x = np.concatenate([x, out], axis=-1)

    return x