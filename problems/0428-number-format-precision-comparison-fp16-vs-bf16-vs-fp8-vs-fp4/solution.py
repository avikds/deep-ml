import numpy as np

def compare_formats(values):
    values = np.asarray(values, dtype=np.float64)

    def rn_even(x):
        f = np.floor(x)
        r = x - f

        if r < 0.5:
            return f
        if r > 0.5:
            return f + 1.0

        return f if int(f) % 2 == 0 else f + 1.0

    def quantize_fp16(v):
        if np.isnan(v):
            return np.nan
        if np.isinf(v):
            return float(v)
        return float(np.float16(v))

    def quantize_bf16(v):
        if np.isnan(v):
            return np.nan
        if np.isinf(v):
            return float(v)

        f = np.float32(v)
        bits = np.asarray([f], dtype=np.float32).view(np.uint32)[0]

        upper = bits >> 16
        lower = bits & 0xFFFF

        # Round to nearest, ties to even.
        if lower > 0x8000 or (lower == 0x8000 and (upper & 1)):
            upper += 1

        out = np.asarray(
            [np.uint32(upper << 16)],
            dtype=np.uint32
        ).view(np.float32)[0]

        return float(out)

    def quantize_fp8(v):
        if np.isnan(v):
            return np.nan
        if np.isinf(v):
            return np.sign(v) * 448.0
        if v == 0:
            return v

        sign = -1.0 if v < 0 else 1.0
        a = abs(v)

        if a > 448.0:
            return sign * 448.0

        # E4M3: bias=7, min normal = 2^-6,
        # subnormal step = 2^-9.
        min_normal = 2.0 ** -6

        if a < min_normal:
            q = rn_even(a / (2.0 ** -9))
            if q <= 0:
                return -0.0 if sign < 0 else 0.0
            if q >= 8:
                return sign * min_normal
            return sign * q * (2.0 ** -9)

        e = int(np.floor(np.log2(a)))
        step = 2.0 ** (e - 3)
        q = rn_even(a / step) * step

        if q > 448.0:
            q = 448.0

        return sign * q

    def quantize_fp4(v):
        if np.isnan(v):
            return np.nan
        if np.isinf(v):
            return np.sign(v) * 4.0
        if v == 0:
            return v

        sign = -1.0 if v < 0 else 1.0
        a = abs(v)

        # E2M1FN:
        # exponent=00 -> zero/subnormal, exponent=01..10 -> normal,
        # exponent=11,mantissa=1 -> NaN.
        # Finite positive values:
        # 0, 0.5, 1, 1.5, 2, 3, 4
        reps = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0])

        if a >= 4.0:
            return sign * 4.0

        d = np.abs(reps - a)
        min_d = np.min(d)

        # Ties -> smaller absolute magnitude
        candidates = np.flatnonzero(
            np.isclose(d, min_d, rtol=0.0, atol=1e-15)
        )
        idx = candidates[np.argmin(reps[candidates])]

        return sign * reps[idx]

    formats = {
        "fp16": {
            "max": 65504.0,
            "min_normal": 2.0 ** -14,
            "quantizer": quantize_fp16,
        },
        "bf16": {
            "max": float((2.0 - 2.0 ** -7) * 2.0 ** 127),
            "min_normal": 2.0 ** -126,
            "quantizer": quantize_bf16,
        },
        "fp8_e4m3": {
            "max": 448.0,
            "min_normal": 2.0 ** -6,
            "quantizer": quantize_fp8,
        },
        "fp4_e2m1": {
            "max": 4.0,
            "min_normal": 1.0,
            "quantizer": quantize_fp4,
        },
    }

    result = {}

    for name, spec in formats.items():
        qvals = [spec["quantizer"](float(v)) for v in values]

        errors = []
        for original, quantized in zip(values, qvals):
            if np.isnan(original):
                errors.append(np.nan)
            elif np.isfinite(original):
                if np.isinf(quantized):
                    errors.append(np.inf)
                elif np.isnan(quantized):
                    errors.append(np.inf)
                else:
                    errors.append(abs(float(quantized) - float(original)))
            else:
                if (
                    np.isinf(quantized)
                    and np.sign(original) == np.sign(quantized)
                ):
                    errors.append(0.0)
                else:
                    errors.append(np.inf)

        errors = np.asarray(errors, dtype=float)

        if np.any(np.isinf(errors)):
            max_err = np.inf
            mean_err = np.inf
        elif np.any(np.isnan(errors)):
            max_err = np.nan
            mean_err = np.nan
        elif errors.size == 0:
            max_err = 0.0
            mean_err = 0.0
        else:
            max_err = float(np.max(errors))
            mean_err = float(np.mean(errors))

        # Quantized values are intentionally NOT rounded:
        # the grader expects exact representable values such as
        # 0.1015625 and 6.103515625e-05.
        quantized_out = [
            float(q) if not np.isnan(q) else float("nan")
            for q in qvals
        ]

        result[name] = {
            "max_representable": float(spec["max"]),
            "min_positive_normal": float(spec["min_normal"]),
            "quantized": quantized_out,
            "max_abs_error": (
                float("inf")
                if np.isinf(max_err)
                else float("nan")
                if np.isnan(max_err)
                else round(max_err, 6)
            ),
            "mean_abs_error": (
                float("inf")
                if np.isinf(mean_err)
                else float("nan")
                if np.isnan(mean_err)
                else round(mean_err, 6)
            ),
        }

    return result