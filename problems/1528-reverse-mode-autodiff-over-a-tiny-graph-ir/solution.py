import numpy as np


def forward(node, bufs, cache):
    if node in cache:
        return cache[node]

    op = node[0]

    if op == "buf":
        out = np.asarray(bufs[node[1]], dtype=np.float32)

    elif op == "const":
        out = np.asarray(node[1], dtype=np.float32)

    elif op == "add":
        out = forward(node[1], bufs, cache) + forward(node[2], bufs, cache)

    elif op == "mul":
        out = forward(node[1], bufs, cache) * forward(node[2], bufs, cache)

    elif op == "neg":
        out = -forward(node[1], bufs, cache)

    elif op == "exp":
        out = np.exp(forward(node[1], bufs, cache))

    elif op == "recip":
        out = 1.0 / forward(node[1], bufs, cache)

    elif op == "max":
        a = forward(node[1], bufs, cache)
        b = forward(node[2], bufs, cache)
        out = np.maximum(a, b)

    elif op == "reshape":
        out = np.reshape(forward(node[1], bufs, cache), node[2])

    elif op == "expand":
        out = np.broadcast_to(forward(node[1], bufs, cache), node[2])

    elif op == "permute":
        out = np.transpose(forward(node[1], bufs, cache), node[2])

    elif op == "sum":
        out = np.sum(forward(node[1], bufs, cache), axis=node[2], keepdims=True)

    else:
        raise ValueError(f"Unknown op: {op}")

    out = np.asarray(out, dtype=np.float32)
    cache[node] = out
    return out


def _children(node):
    op = node[0]

    if op in ("buf", "const"):
        return ()

    if op in ("add", "mul", "max"):
        return (node[1], node[2])

    return (node[1],)


def _topo(node):
    """Return unique nodes in postorder."""
    order = []
    seen = set()
    stack = [(node, False)]

    while stack:
        cur, expanded = stack.pop()

        if cur in seen and not expanded:
            continue

        if expanded:
            order.append(cur)
            continue

        seen.add(cur)
        stack.append((cur, True))

        for child in reversed(_children(cur)):
            if child not in seen:
                stack.append((child, False))

    return order


def backward(loss, bufs, wrt):
    cache = {}
    order = _topo(loss)

    # Evaluate all reachable expressions.
    forward(loss, bufs, cache)

    grads = {}

    def add_grad(node, g):
        g = np.asarray(g, dtype=np.float32)
        if node in grads:
            grads[node] += g
        else:
            grads[node] = g.copy()

    def reduce_to_shape(g, shape):
        g = np.asarray(g, dtype=np.float32)

        while g.ndim > len(shape):
            g = np.sum(g, axis=0)

        for axis, size in enumerate(shape):
            if size == 1 and g.shape[axis] != 1:
                g = np.sum(g, axis=axis, keepdims=True)

        return g.reshape(shape)

    add_grad(loss, np.ones_like(cache[loss], dtype=np.float32))

    for node in reversed(order):
        if node not in grads:
            continue

        g = grads[node]
        op = node[0]

        if op in ("buf", "const"):
            continue

        if op == "add":
            a, b = node[1], node[2]
            add_grad(a, reduce_to_shape(g, cache[a].shape))
            add_grad(b, reduce_to_shape(g, cache[b].shape))

        elif op == "mul":
            a, b = node[1], node[2]
            add_grad(a, reduce_to_shape(g * cache[b], cache[a].shape))
            add_grad(b, reduce_to_shape(g * cache[a], cache[b].shape))

        elif op == "neg":
            add_grad(node[1], -g)

        elif op == "exp":
            add_grad(node[1], g * cache[node])

        elif op == "recip":
            out = cache[node]
            add_grad(node[1], -g * out * out)

        elif op == "max":
            a, b = node[1], node[2]
            va, vb = cache[a], cache[b]

            # Ties go to the first operand.
            mask_a = va >= vb
            mask_b = va < vb

            add_grad(a, reduce_to_shape(g * mask_a, va.shape))
            add_grad(b, reduce_to_shape(g * mask_b, vb.shape))

        elif op == "reshape":
            a = node[1]
            add_grad(a, g.reshape(cache[a].shape))

        elif op == "expand":
            a = node[1]
            input_shape = cache[a].shape
            output_shape = cache[node].shape

            padded = (1,) * (len(output_shape) - len(input_shape)) + input_shape
            axes = [
                i
                for i, (si, so) in enumerate(zip(padded, output_shape))
                if si == 1 and so != 1
            ]

            reduced = g
            for axis in reversed(axes):
                reduced = np.sum(reduced, axis=axis, keepdims=True)

            add_grad(a, reduced.reshape(input_shape))

        elif op == "permute":
            a = node[1]
            perm = tuple(node[2])
            inverse = np.argsort(perm)
            add_grad(a, np.transpose(g, inverse))

        elif op == "sum":
            a = node[1]
            add_grad(a, np.broadcast_to(g, cache[a].shape))

        else:
            raise ValueError(f"Unknown op: {op}")

    result = {}
    for name in wrt:
        node = ("buf", name)

        if node in grads:
            result[name] = np.asarray(grads[node], dtype=np.float32)
        else:
            result[name] = np.zeros_like(
                np.asarray(bufs[name], dtype=np.float32)
            )

    return result


def numeric_gradient(loss, bufs, name, eps=1e-3):
    local_bufs = {
        k: np.asarray(v, dtype=np.float64).copy()
        for k, v in bufs.items()
    }

    x = local_bufs[name]
    grad = np.zeros_like(x, dtype=np.float64)

    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])

    while not it.finished:
        idx = it.multi_index
        old = x[idx]

        x[idx] = old + eps
        plus = forward(loss, local_bufs, {})
        plus_val = float(np.asarray(plus).reshape(-1)[0])

        x[idx] = old - eps
        minus = forward(loss, local_bufs, {})
        minus_val = float(np.asarray(minus).reshape(-1)[0])

        x[idx] = old
        grad[idx] = (plus_val - minus_val) / (2.0 * eps)

        it.iternext()

    return grad