import math

def minifloat_decode(bits, e, m, reserve_top=True):
    bits = int(bits)
    sign = -1.0 if (bits >> (e + m)) & 1 else 1.0

    exp_mask = (1 << e) - 1
    mant_mask = (1 << m) - 1

    exp = (bits >> m) & exp_mask
    mant = bits & mant_mask
    bias = (1 << (e - 1)) - 1

    if reserve_top and exp == exp_mask:
        if mant == 0:
            return math.copysign(float("inf"), sign)
        return float("nan")

    if not reserve_top and bits == (1 << (1 + e + m)) - 1:
        return float("nan")

    if exp == 0:
        if mant == 0:
            return math.copysign(0.0, sign)
        value = (mant / (1 << m)) * (2.0 ** (1 - bias))
    else:
        value = (1.0 + mant / (1 << m)) * (2.0 ** (exp - bias))

    return sign * value


def max_normal(e, m, reserve_top=True):
    bias = (1 << (e - 1)) - 1
    exp = (1 << e) - 2 if reserve_top else (1 << e) - 1
    mant = (1 << m) - 1

    return (1.0 + mant / (1 << m)) * (2.0 ** (exp - bias))


def min_subnormal(e, m):
    bias = (1 << (e - 1)) - 1
    return 2.0 ** (1 - bias - m)


def minifloat_encode(x, e, m, reserve_top=True):
    x = float(x)

    if x == 0.0:
        return 0

    if not math.isfinite(x):
        raise ValueError("Input must be finite")

    sign_bit = (1 << (e + m)) if x < 0 else 0
    target = abs(x)

    max_value = max_normal(e, m, reserve_top)

    if target >= max_value:
        return sign_bit | ((1 << e) - (2 if reserve_top else 1)) << m | ((1 << m) - 1)

    best_bits = 0
    best_error = float("inf")
    best_even = False

    for exp in range(1 << e):
        for mant in range(1 << m):
            bits = sign_bit | (exp << m) | mant
            value = minifloat_decode(bits, e, m, reserve_top)

            if not math.isfinite(value):
                continue

            error = abs(abs(value) - target)
            even = (mant % 2 == 0)

            if error < best_error or (error == best_error and even and not best_even):
                best_error = error
                best_bits = bits
                best_even = even

    return best_bits


def relative_error(x, e, m, reserve_top=True):
    x = float(x)

    if x == 0.0:
        return 0.0

    bits = minifloat_encode(x, e, m, reserve_top)
    decoded = minifloat_decode(bits, e, m, reserve_top)

    return round(abs(decoded - x) / abs(x), 6)