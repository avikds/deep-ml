import numpy as np


def shape_after(shape0, ops):
    shape = tuple(shape0)

    for op in ops:
        kind = op[0]

        if kind == "reshape":
            shape = tuple(op[1])

        elif kind == "expand":
            new_shape = tuple(op[1])
            if len(new_shape) < len(shape):
                raise ValueError("expand cannot reduce rank")

            offset = len(new_shape) - len(shape)
            for i, s in enumerate(shape):
                target = new_shape[offset + i]
                if s != 1 and s != target:
                    raise ValueError("incompatible expand shape")
            shape = new_shape

        elif kind == "permute":
            perm = tuple(op[1])
            shape = tuple(shape[i] for i in perm)

        elif kind == "flip":
            # Shape is unchanged.
            shape = shape

        elif kind == "pad":
            pads = op[1]
            shape = tuple(
                s + int(p[0]) + int(p[1])
                for s, p in zip(shape, pads)
            )

        elif kind == "shrink":
            ranges = op[1]
            shape = tuple(
                int(end) - int(start)
                for start, end in ranges
            )

        else:
            raise ValueError(f"unknown op: {kind}")

    return shape


def source_index(shape0, ops, idx):
    # Save every intermediate shape so each reverse step knows
    # both its input and output shape.
    shapes = [tuple(shape0)]
    for op in ops:
        shapes.append(shape_after(shapes[-1], [op]))

    cur_idx = tuple(int(i) for i in idx)
    valid = True

    # Walk the movement chain backwards.
    for k in range(len(ops) - 1, -1, -1):
        op = ops[k]
        kind = op[0]
        prev_shape = shapes[k]
        cur_shape = shapes[k + 1]

        if kind == "reshape":
            # Flatten in the current shape, then unflatten in prev_shape.
            flat = 0
            stride = 1
            for i in range(len(cur_shape) - 1, -1, -1):
                flat += cur_idx[i] * stride
                stride *= cur_shape[i]

            new_idx = []
            for i, size in enumerate(prev_shape):
                stride = int(np.prod(prev_shape[i + 1:], dtype=np.int64))
                new_idx.append((flat // stride) % size)

            cur_idx = tuple(new_idx)

        elif kind == "expand":
            # Right-align the old shape with the expanded shape.
            offset = len(cur_shape) - len(prev_shape)
            new_idx = []

            for i, size in enumerate(prev_shape):
                out_i = i + offset
                if size == 1:
                    new_idx.append(0)
                else:
                    new_idx.append(cur_idx[out_i])

            cur_idx = tuple(new_idx)

        elif kind == "permute":
            perm = tuple(op[1])
            inverse = np.argsort(perm)
            cur_idx = tuple(cur_idx[i] for i in inverse)

        elif kind == "flip":
            axes = op[1]
            if isinstance(axes, int):
                axes = (axes,)

            axes = {
                a if a >= 0 else len(prev_shape) + a
                for a in axes
            }

            cur_idx = tuple(
                prev_shape[i] - 1 - cur_idx[i]
                if i in axes else cur_idx[i]
                for i in range(len(prev_shape))
            )

        elif kind == "pad":
            pads = op[1]
            new_idx = []

            for i, (low, high) in enumerate(pads):
                v = cur_idx[i] - int(low)
                if v < 0 or v >= prev_shape[i]:
                    valid = False
                new_idx.append(v)

            cur_idx = tuple(new_idx)

        elif kind == "shrink":
            ranges = op[1]
            cur_idx = tuple(
                cur_idx[i] + int(ranges[i][0])
                for i in range(len(prev_shape))
            )

    # Compute the original row-major flat offset even when invalid.
    flat_offset = 0
    stride = 1
    for i in range(len(shape0) - 1, -1, -1):
        flat_offset += cur_idx[i] * stride
        stride *= shape0[i]

    return int(flat_offset), bool(valid)


def materialize(shape0, ops, buf):
    buf = np.asarray(buf)
    out_shape = shape_after(shape0, ops)

    out = np.empty(out_shape, dtype=buf.dtype)

    for idx in np.ndindex(out_shape):
        flat, valid = source_index(shape0, ops, idx)

        if valid:
            out[idx] = buf.reshape(-1)[flat]
        else:
            out[idx] = 0.0

    return out