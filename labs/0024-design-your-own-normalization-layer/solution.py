import numpy as np

def normalize(x):
    '''
    Apply RMSNorm-style normalization across the last dimension.
    '''
    # Convert to floating point for safe division and computation.
    x = np.asarray(x, dtype=np.float64)

    # Root mean square over the feature dimension.
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + 1e-8)

    # Normalize each feature vector.
    result = x / rms

    # Ensure finite output.
    result = np.nan_to_num(
        result,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return result