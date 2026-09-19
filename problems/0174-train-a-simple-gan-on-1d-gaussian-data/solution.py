import numpy as np

def train_gan(
    mean_real: float,
    std_real: float,
    latent_dim: int = 1,
    hidden_dim: int = 16,
    learning_rate: float = 0.001,
    epochs: int = 5000,
    batch_size: int = 128,
    seed: int = 42
):
    rng = np.random.RandomState(seed)

    # Tiny generator: z -> ReLU -> linear
    Wg1 = rng.randn(latent_dim, hidden_dim) * 0.01
    bg1 = np.zeros(hidden_dim)
    Wg2 = rng.randn(hidden_dim, 1) * 0.01
    bg2 = np.zeros(1)

    # Discriminator: x -> ReLU -> sigmoid
    Wd1 = rng.randn(1, hidden_dim) * 0.01
    bd1 = np.zeros(hidden_dim)
    Wd2 = rng.randn(hidden_dim, 1) * 0.01
    bd2 = np.zeros(1)

    eps = 1e-8

    def sigmoid(x):
        x = np.clip(x, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-x))

    def generator(z, cache=False):
        z = np.asarray(z, dtype=float)
        h_pre = z @ Wg1 + bg1
        h = np.maximum(h_pre, 0.0)
        out = h @ Wg2 + bg2

        if cache:
            return out, h_pre, h
        return out

    def discriminator(x, cache=False):
        h_pre = x @ Wd1 + bd1
        h = np.maximum(h_pre, 0.0)
        logits = h @ Wd2 + bd2
        p = sigmoid(logits)

        if cache:
            return p, h_pre, h
        return p

    for _ in range(epochs):
        # ---------------- Discriminator ----------------
        x_real = rng.normal(
            mean_real, std_real, size=(batch_size, 1)
        )
        z = rng.normal(
            0.0, 1.0, size=(batch_size, latent_dim)
        )

        x_fake, hg_pre, hg = generator(z, cache=True)

        p_real, hd_r_pre, hd_r = discriminator(x_real, cache=True)
        p_fake, hd_f_pre, hd_f = discriminator(x_fake, cache=True)

        # BCE gradients
        dLr = (p_real - 1.0) / batch_size
        dLf = p_fake / batch_size

        dWd2 = hd_r.T @ dLr + hd_f.T @ dLf
        dbd2 = np.sum(dLr, axis=0) + np.sum(dLf, axis=0)

        dh_r = (dLr @ Wd2.T) * (hd_r_pre > 0)
        dh_f = (dLf @ Wd2.T) * (hd_f_pre > 0)

        dWd1 = x_real.T @ dh_r + x_fake.T @ dh_f
        dbd1 = np.sum(dh_r, axis=0) + np.sum(dh_f, axis=0)

        Wd2 -= learning_rate * dWd2
        bd2 -= learning_rate * dbd2
        Wd1 -= learning_rate * dWd1
        bd1 -= learning_rate * dbd1

        # ---------------- Generator ----------------
        z = rng.normal(
            0.0, 1.0, size=(batch_size, latent_dim)
        )

        x_fake, hg_pre, hg = generator(z, cache=True)
        p_fake, hd_f_pre, hd_f = discriminator(x_fake, cache=True)

        # Non-saturating loss: -mean(log(D(G(z))))
        dL = (p_fake - 1.0) / batch_size

        dh = (dL @ Wd2.T) * (hd_f_pre > 0)
        dx = dh @ Wd1.T

        dWg2 = hg.T @ dx
        dbg2 = np.sum(dx, axis=0)

        dhg = (dx @ Wg2.T) * (hg_pre > 0)
        dWg1 = z.T @ dhg
        dbg1 = np.sum(dhg, axis=0)

        Wg2 -= learning_rate * dWg2
        bg2 -= learning_rate * dbg2
        Wg1 -= learning_rate * dWg1
        bg1 -= learning_rate * dbg1

        # Keep the tiny generator numerically stable.
        Wg1 = np.clip(Wg1, -0.1, 0.1)
        Wg2 = np.clip(Wg2, -0.1, 0.1)
        bg1 = np.clip(bg1, -0.1, 0.1)
        bg2 = np.clip(bg2, -0.1, 0.1)

    def gen_forward(z):
        z = np.asarray(z, dtype=float)

        if z.ndim == 1:
            z = z.reshape(-1, latent_dim)

        h = np.maximum(z @ Wg1 + bg1, 0.0)
        x_gen = h @ Wg2 + bg2

        return x_gen, h, z

    return gen_forward