import numpy as np

def conv3d_forward_pass(
    input_volume: np.ndarray,
    kernel: np.ndarray,
    stride: tuple[int, int, int] = (1, 1, 1),
    padding: tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    """
    Perform a 3D convolution forward pass.
    """
    input_volume = np.asarray(input_volume, dtype=float)
    kernel = np.asarray(kernel, dtype=float)

    if input_volume.ndim != 4:
        raise ValueError("input_volume must have shape (C, D, H, W)")

    if kernel.ndim != 4:
        raise ValueError("kernel must have shape (C, kD, kH, kW)")

    C, D, H, W = input_volume.shape
    kC, kD, kH, kW = kernel.shape

    if C != kC:
        raise ValueError("Kernel channel count must match input channels")

    sd, sh, sw = stride
    pd, ph, pw = padding

    if sd <= 0 or sh <= 0 or sw <= 0:
        raise ValueError("Stride values must be positive")

    if pd < 0 or ph < 0 or pw < 0:
        raise ValueError("Padding values must be non-negative")

    # Zero-pad: (C, D, H, W)
    padded = np.pad(
        input_volume,
        ((0, 0), (pd, pd), (ph, ph), (pw, pw)),
        mode="constant"
    )

    Dp, Hp, Wp = padded.shape[1:]

    D_out = (Dp - kD) // sd + 1
    H_out = (Hp - kH) // sh + 1
    W_out = (Wp - kW) // sw + 1

    if D_out <= 0 or H_out <= 0 or W_out <= 0:
        raise ValueError("Kernel is larger than the padded input")

    output = np.zeros(
        (1, D_out, H_out, W_out),
        dtype=float
    )

    for d in range(D_out):
        d_start = d * sd

        for h in range(H_out):
            h_start = h * sh

            for w in range(W_out):
                w_start = w * sw

                patch = padded[
                    :,
                    d_start:d_start + kD,
                    h_start:h_start + kH,
                    w_start:w_start + kW
                ]

                output[0, d, h, w] = np.sum(patch * kernel)

    return output