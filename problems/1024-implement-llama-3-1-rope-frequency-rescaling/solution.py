import numpy as np
import math

def llama3_rope_rescale(
    inv_freq,
    original_context_length,
    low_freq_factor,
    high_freq_factor,
    scaling_factor
):
    """
    Apply Llama 3.1 style piecewise rescaling to RoPE inverse frequencies.
    """
    inv_freq = np.asarray(inv_freq, dtype=float)

    low_freq_wavelen = original_context_length / low_freq_factor
    high_freq_wavelen = original_context_length / high_freq_factor

    result = inv_freq.copy()

    for i, freq in enumerate(inv_freq):
        wavelen = 2.0 * math.pi / freq

        if wavelen > low_freq_wavelen:
            # Low frequency: stretch by scaling factor.
            result[i] = freq / scaling_factor

        elif wavelen < high_freq_wavelen:
            # High frequency: leave unchanged.
            result[i] = freq

        else:
            # Medium frequency: smooth interpolation.
            smooth = (
                (original_context_length / wavelen - low_freq_factor)
                / (high_freq_factor - low_freq_factor)
            )

            result[i] = (
                (1.0 - smooth) * (freq / scaling_factor)
                + smooth * freq
            )

    return np.round(result, 6).tolist()