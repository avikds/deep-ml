import numpy as np

def video_tokenize(video, codebook, encoder_weight, patch_size):
    """
    Tokenize a video into a grid of discrete codebook indices using a
    spatiotemporal VQ-VAE encoder.

    Args:
        video: array of shape (T, H, W, C).
        codebook: array of shape (K, D).
        encoder_weight: array of shape (pt*ph*pw*C, D).
        patch_size: tuple (pt, ph, pw).

    Returns:
        Nested list of codebook indices, or -1 if the inputs are invalid.
    """
    try:
        video = np.asarray(video)
        codebook = np.asarray(codebook)
        encoder_weight = np.asarray(encoder_weight)

        # Basic shape validation.
        if video.ndim != 4:
            return -1
        if codebook.ndim != 2:
            return -1
        if encoder_weight.ndim != 2:
            return -1

        if len(patch_size) != 3:
            return -1

        pt, ph, pw = patch_size

        if not all(isinstance(x, (int, np.integer)) for x in (pt, ph, pw)):
            return -1
        if pt <= 0 or ph <= 0 or pw <= 0:
            return -1

        T, H, W, C = video.shape
        K, D = codebook.shape

        if K == 0 or D == 0:
            return -1

        # Video dimensions must be exactly divisible by patch dimensions.
        if T % pt != 0 or H % ph != 0 or W % pw != 0:
            return -1

        # Encoder input/output dimensions must match the patch and codebook.
        expected_in_dim = pt * ph * pw * C
        if encoder_weight.shape[0] != expected_in_dim:
            return -1
        if encoder_weight.shape[1] != D:
            return -1

        n_t = T // pt
        n_h = H // ph
        n_w = W // pw

        tokens = []

        for ti in range(n_t):
            row_tokens = []

            for hi in range(n_h):
                col_tokens = []

                for wi in range(n_w):
                    patch = video[
                        ti * pt:(ti + 1) * pt,
                        hi * ph:(hi + 1) * ph,
                        wi * pw:(wi + 1) * pw,
                        :
                    ]

                    # Standard C-order flattening.
                    flat_patch = patch.reshape(-1)

                    # Linear encoder projection.
                    latent = flat_patch @ encoder_weight

                    # Nearest codebook entry.
                    distances = np.sum(
                        (codebook - latent[None, :]) ** 2,
                        axis=1
                    )

                    # argmin picks the lowest index on ties.
                    token = int(np.argmin(distances))
                    col_tokens.append(token)

                row_tokens.append(col_tokens)

            tokens.append(row_tokens)

        return tokens

    except (TypeError, ValueError, IndexError):
        return -1