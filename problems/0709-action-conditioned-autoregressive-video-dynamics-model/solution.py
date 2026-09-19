import numpy as np

def rollout_video_dynamics(
    z0: np.ndarray,
    actions: np.ndarray,
    W_scale: np.ndarray,
    b_scale: np.ndarray,
    W_shift: np.ndarray,
    b_shift: np.ndarray,
    W_hidden: np.ndarray,
    b_hidden: np.ndarray,
    W_out: np.ndarray,
    b_out: np.ndarray
) -> np.ndarray:
    """
    Autoregressively roll out a video dynamics model conditioned on actions.
    """
    z = np.asarray(z0, dtype=float).copy()
    actions = np.asarray(actions, dtype=float)

    W_scale = np.asarray(W_scale, dtype=float)
    b_scale = np.asarray(b_scale, dtype=float)
    W_shift = np.asarray(W_shift, dtype=float)
    b_shift = np.asarray(b_shift, dtype=float)
    W_hidden = np.asarray(W_hidden, dtype=float)
    b_hidden = np.asarray(b_hidden, dtype=float)
    W_out = np.asarray(W_out, dtype=float)
    b_out = np.asarray(b_out, dtype=float)

    T = actions.shape[0]
    D = z.shape[0]

    rollout = np.zeros((T, D), dtype=float)

    for t in range(T):
        action = actions[t]

        # Action-conditioned scale and shift.
        scale = action @ W_scale + b_scale
        shift = action @ W_shift + b_shift

        # Modulate current latent.
        z_mod = scale * z + shift

        # Hidden transformation with tanh.
        h = np.tanh(W_hidden @ z_mod + b_hidden)

        # Output transformation.
        update = W_out @ h + b_out

        # Residual next-state prediction.
        z = z + update
        rollout[t] = z

    return rollout