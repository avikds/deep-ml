import numpy as np

def categorical_projection(current_probs, next_probs, reward, gamma, done, v_min, v_max, n_atoms):
    """
    Perform categorical distributional Bellman projection and compute cross-entropy loss.
    """
    current_probs = np.asarray(current_probs, dtype=float)
    next_probs = np.asarray(next_probs, dtype=float)

    # Fixed support.
    support = np.linspace(v_min, v_max, n_atoms)
    delta_z = (v_max - v_min) / (n_atoms - 1)

    projected = np.zeros(n_atoms, dtype=float)

    # Bellman-transformed support.
    if done:
        tz = np.full(n_atoms, reward, dtype=float)
    else:
        tz = reward + gamma * support

    tz = np.clip(tz, v_min, v_max)

    # Fractional location of each transformed atom.
    b = (tz - v_min) / delta_z
    lower = np.floor(b).astype(int)
    upper = np.ceil(b).astype(int)

    for j in range(n_atoms):
        l = lower[j]
        u = upper[j]

        if l == u:
            projected[l] += next_probs[j]
        else:
            projected[l] += next_probs[j] * (u - b[j])
            projected[u] += next_probs[j] * (b[j] - l)

    # Cross-entropy loss.
    loss = -np.sum(
        projected * np.log(np.clip(current_probs, 1e-8, None))
    )

    return projected, float(loss)