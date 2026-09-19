import numpy as np

def xavier_init(
    fan_in: int,
    fan_out: int,
    mode: str = 'uniform',
    seed: int = 42
) -> dict:
    """
    Perform Xavier/Glorot weight initialization.
    """
    if mode not in ('uniform', 'normal'):
        raise ValueError("mode must be 'uniform' or 'normal'")

    if fan_in <= 0 or fan_out <= 0:
        raise ValueError("fan_in and fan_out must be positive")

    rng = np.random.RandomState(seed)

    if mode == 'uniform':
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        weights = rng.uniform(
            -limit, limit, size=(fan_in, fan_out)
        )
        param = limit
    else:
        std = np.sqrt(2.0 / (fan_in + fan_out))
        weights = rng.normal(
            0.0, std, size=(fan_in, fan_out)
        )
        param = std

    return {
        'weights': np.round(weights, 4).tolist(),
        'shape': [fan_in, fan_out],
        'param': round(float(param), 4)
    }