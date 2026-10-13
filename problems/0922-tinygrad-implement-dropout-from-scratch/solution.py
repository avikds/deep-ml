from tinygrad import Tensor

def dropout(x, p, training):
    # No dropout during evaluation or when p = 0.
    if not training or p == 0:
        return x

    # Keep each element independently with probability 1 - p.
    noise = Tensor.rand(*x.shape)
    mask = (noise >= p).cast(x.dtype)

    # Inverted dropout preserves the expected activation magnitude.
    return x * mask / (1.0 - p)