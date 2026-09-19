import numpy as np

def hashed_tile_coding(
    state: list,
    num_tilings: int,
    num_tiles_per_dim: list,
    state_lows: list,
    state_highs: list,
    memory_size: int,
    weights: np.ndarray = None
) -> tuple:
    """
    Compute memory-efficient tile coding features using hashing.
    """
    state = np.asarray(state, dtype=float)
    num_tiles_per_dim = np.asarray(num_tiles_per_dim, dtype=int)
    state_lows = np.asarray(state_lows, dtype=float)
    state_highs = np.asarray(state_highs, dtype=float)

    primes = [509, 521, 523, 541, 547, 557, 563, 569, 571, 577]

    tile_widths = (state_highs - state_lows) / num_tiles_per_dim

    active_tiles = []

    for t in range(num_tilings):
        offsets = t * tile_widths / num_tilings

        coords = np.floor(
            (state - state_lows + offsets) / tile_widths
        ).astype(int)

        key = [t] + coords.tolist()

        h = sum(
            coord * primes[i % len(primes)]
            for i, coord in enumerate(key)
        )

        active_tiles.append(int(abs(h) % memory_size))

    active_tiles.sort()

    if weights is None:
        value_estimate = 0.0
    else:
        weights = np.asarray(weights)
        value_estimate = float(np.sum(weights[active_tiles]))
        value_estimate = round(value_estimate, 4)

    return active_tiles, value_estimate