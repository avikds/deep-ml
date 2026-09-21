import numpy as np

def llama3_rope_inv_freq(
    dim,
    base=500000.0,
    factor=8.0,
    low_freq_factor=1.0,
    high_freq_factor=4.0,
    original_context_length=8192
):
    """
    Compute RoPE inverse frequencies with Llama 3.1-style frequency scaling.
    """
    # Base inverse frequencies for i = 0 .. dim/2 - 1.
    i = np.arange(dim // 2, dtype=float)
    inv_freq = 1.0 / (base ** (2.0 * i / dim))

    # Wavelength of each frequency.
    wavelengths = 2.0 * np.pi / inv_freq

    low_freq_wavelen = original_context_length / low_freq_factor
    high_freq_wavelen = original_context_length / high_freq_factor

    result = inv_freq.copy()

    # High-frequency region: unchanged.
    high_mask = wavelengths < high_freq_wavelen

    # Low-frequency region: divide by factor.
    low_mask = wavelengths > low_freq_wavelen

    # Medium-frequency region: linear interpolation.
    mid_mask = ~(high_mask | low_mask)

    smooth = (
        (original_context_length / wavelengths[mid_mask] - low_freq_factor)
        / (high_freq_factor - low_freq_factor)
    )

    result[mid_mask] = (
        (1.0 - smooth) * (inv_freq[mid_mask] / factor)
        + smooth * inv_freq[mid_mask]
    )

    result[low_mask] = inv_freq[low_mask] / factor

    return np.round(result, 8).tolist()