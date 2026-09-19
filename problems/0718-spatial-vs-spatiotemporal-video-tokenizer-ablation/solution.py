import numpy as np

def video_tokenizer_reconstruct(video, patch_t: int, patch_h: int, patch_w: int, mode: str):
    """
    Reconstruct a video after mean-pool tokenization.

    Args:
        video: array-like of shape (T, H, W, C)
        patch_t: temporal patch size (ignored when mode == 'spatial')
        patch_h: spatial patch height
        patch_w: spatial patch width
        mode: 'spatial' or 'spatiotemporal'

    Returns:
        Reconstructed video as a nested list of shape (T, H, W, C).
    """
    video = np.asarray(video, dtype=float)

    if video.ndim != 4:
        raise ValueError("video must have shape (T, H, W, C)")

    if mode not in ("spatial", "spatiotemporal"):
        raise ValueError("mode must be 'spatial' or 'spatiotemporal'")

    T, H, W, C = video.shape

    # Spatial mode always uses a temporal patch size of 1.
    pt = 1 if mode == "spatial" else patch_t

    if pt <= 0 or patch_h <= 0 or patch_w <= 0:
        raise ValueError("patch sizes must be positive")

    if T % pt != 0 or H % patch_h != 0 or W % patch_w != 0:
        raise ValueError("video dimensions must be divisible by patch sizes")

    reconstructed = np.empty_like(video, dtype=float)

    for t in range(0, T, pt):
        for h in range(0, H, patch_h):
            for w in range(0, W, patch_w):
                patch = video[
                    t:t + pt,
                    h:h + patch_h,
                    w:w + patch_w,
                    :
                ]

                # Mean per channel over all temporal/spatial positions.
                mean = np.mean(patch, axis=(0, 1, 2), keepdims=True)

                # Broadcast the per-channel mean over the entire patch.
                reconstructed[
                    t:t + pt,
                    h:h + patch_h,
                    w:w + patch_w,
                    :
                ] = mean

    return reconstructed.tolist()