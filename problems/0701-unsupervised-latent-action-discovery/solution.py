import numpy as np

def latent_action_forward(
    obs_t: np.ndarray,
    obs_tp1: np.ndarray,
    W_enc: np.ndarray,
    b_enc: np.ndarray,
    codebook: np.ndarray,
    W_dec: np.ndarray,
    b_dec: np.ndarray
) -> np.ndarray:
    """
    Forward pass for unsupervised latent action discovery.

    Encodes (o_t, o_{t+1}) pairs into a continuous latent, snaps the latent
    to the nearest codebook vector, and decodes the predicted next observation.

    Returns:
        Predicted next observations of shape (B, D_obs).
    """
    obs_t = np.asarray(obs_t, dtype=float)
    obs_tp1 = np.asarray(obs_tp1, dtype=float)
    W_enc = np.asarray(W_enc, dtype=float)
    b_enc = np.asarray(b_enc, dtype=float)
    codebook = np.asarray(codebook, dtype=float)
    W_dec = np.asarray(W_dec, dtype=float)
    b_dec = np.asarray(b_dec, dtype=float)

    # 1. Inverse-dynamics encoder.
    encoder_input = np.concatenate([obs_t, obs_tp1], axis=1)
    z_e = encoder_input @ W_enc + b_enc

    # 2. Discrete bottleneck: nearest codebook vector.
    # argmin deterministically chooses the lowest index on ties.
    distances = np.sum(
        (z_e[:, None, :] - codebook[None, :, :]) ** 2,
        axis=2
    )
    indices = np.argmin(distances, axis=1)
    z_q = codebook[indices]

    # 3. Forward-dynamics decoder.
    decoder_input = np.concatenate([obs_t, z_q], axis=1)
    predictions = decoder_input @ W_dec + b_dec

    return predictions