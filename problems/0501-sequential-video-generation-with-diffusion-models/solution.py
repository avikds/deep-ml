import numpy as np


def sequential_video_diffusion(
    noise_chunks: np.ndarray,
    Wd: np.ndarray,
    alpha_bar: np.ndarray,
    n_context: int,
    n_overlap: int,
) -> np.ndarray:
    """
    Generate video frames sequentially using chunk-based diffusion denoising.
    """
    noise_chunks = np.asarray(noise_chunks, dtype=float)
    Wd = np.asarray(Wd, dtype=float)
    alpha_bar = np.asarray(alpha_bar, dtype=float)

    n_chunks, chunk_frames, H, W = noise_chunks.shape
    n_steps = len(alpha_bar) - 1

    if n_chunks == 0:
        return np.empty((0, H, W), dtype=float)

    # Store fully denoised chunks.
    generated_chunks = []

    for chunk_idx in range(n_chunks):
        # Start this chunk from its supplied noise.
        chunk = noise_chunks[chunk_idx].copy()

        # Context bias is zero for the first chunk.
        if chunk_idx == 0 or n_context <= 0:
            context_bias = np.zeros(H * W, dtype=float)
        else:
            # Take trailing context frames from the previous chunk.
            context_frames = generated_chunks[-1][-n_context:]
            context_bias = np.mean(
                context_frames.reshape(-1, H * W),
                axis=0
            )

        # Reverse diffusion: t = n_steps ... 1.
        for t in range(n_steps, 0, -1):
            alpha_t = alpha_bar[t]
            alpha_prev = alpha_bar[t - 1]

            sqrt_alpha_t = np.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = np.sqrt(1.0 - alpha_t)
            sqrt_alpha_prev = np.sqrt(alpha_prev)
            sqrt_one_minus_alpha_prev = np.sqrt(1.0 - alpha_prev)

            for frame_idx in range(chunk_frames):
                x = chunk[frame_idx].reshape(-1)

                # Denoiser prediction.
                eps_pred = Wd @ x + context_bias

                # Predict the clean sample x_0.
                x0_pred = (
                    x - sqrt_one_minus_alpha_t * eps_pred
                ) / sqrt_alpha_t

                # Deterministic DDIM-style update.
                x_prev = (
                    sqrt_alpha_prev * x0_pred
                    + sqrt_one_minus_alpha_prev * eps_pred
                )

                chunk[frame_idx] = x_prev.reshape(H, W)

        generated_chunks.append(chunk)

    # ------------------------------------------------------------
    # Assemble chunks with linear overlap blending.
    # ------------------------------------------------------------
    if n_chunks == 1 or n_overlap == 0:
        return np.concatenate(generated_chunks, axis=0)

    video = generated_chunks[0].copy()

    overlap = min(n_overlap, chunk_frames)

    for chunk_idx in range(1, n_chunks):
        new_chunk = generated_chunks[chunk_idx]

        # Blend the last `overlap` frames of the existing video
        # with the first `overlap` frames of the new chunk.
        for i in range(overlap):
            w_new = (i + 1) / (overlap + 1)
            w_old = 1.0 - w_new

            video[-overlap + i] = (
                w_old * video[-overlap + i]
                + w_new * new_chunk[i]
            )

        # Append the non-overlapping remainder.
        video = np.concatenate(
            [video, new_chunk[overlap:]],
            axis=0
        )

    return video