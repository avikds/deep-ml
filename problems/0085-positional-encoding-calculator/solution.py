import numpy as np

def pos_encoding(position: int, d_model: int):
    # Invalid inputs
    if position == 0 or d_model <= 0:
        return -1

    # Position indices: (position, 1)
    pos = np.arange(position)[:, np.newaxis]

    # Dimension indices: (1, d_model)
    i = np.arange(d_model)[np.newaxis, :]

    # Angle rates from the Transformer paper
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / d_model)

    # Compute angles
    angle_rads = pos * angle_rates

    # Apply sin to even indices and cos to odd indices
    pos_encoding = np.zeros((position, d_model))
    pos_encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])
    pos_encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])

    # Required output type
    pos_encoding = np.float16(pos_encoding)

    return pos_encoding