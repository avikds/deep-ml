import numpy as np

def vq_vae_loss(
    x: np.ndarray,
    z_e: np.ndarray,
    x_recon: np.ndarray,
    codebook: np.ndarray,
    beta: float
) -> float:
    """
    Compute the VQ-VAE training loss.
    """
    x = np.asarray(x, dtype=float)
    z_e = np.asarray(z_e, dtype=float)
    x_recon = np.asarray(x_recon, dtype=float)
    codebook = np.asarray(codebook, dtype=float)

    # Squared Euclidean distance from each encoder output to each codebook entry.
    distances = np.sum(
        (z_e[:, None, :] - codebook[None, :, :]) ** 2,
        axis=2
    )

    # np.argmin deterministically chooses the lowest index in case of ties.
    indices = np.argmin(distances, axis=1)

    # Quantized latent vectors.
    z_q = codebook[indices]

    # Reconstruction loss.
    reconstruction_loss = np.mean((x - x_recon) ** 2)

    # Codebook loss: move selected codebook vectors toward encoder outputs.
    codebook_loss = np.mean((z_q - z_e) ** 2)

    # Commitment loss: encourage encoder outputs to stay close to codebook.
    commitment_loss = np.mean((z_e - z_q) ** 2)

    total_loss = (
        reconstruction_loss
        + codebook_loss
        + beta * commitment_loss
    )

    return float(total_loss)