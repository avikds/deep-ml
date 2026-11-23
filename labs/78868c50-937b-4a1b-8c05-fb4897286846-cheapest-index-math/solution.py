def lower(chain):
    base = tuple(chain["base_shape"])
    ops = chain["ops"]
    out_shape = tuple(chain["out_shape"])

    def add(a, b):
        if type(a) is int and type(b) is int:
            return a + b
        if a == 0:
            return b
        if b == 0:
            return a
        return ("add", a, b)

    def mul(a, b):
        if type(a) is int and type(b) is int:
            return a * b
        if a == 0 or b == 0:
            return 0
        if a == 1:
            return b
        if b == 1:
            return a
        return ("mul", a, b)

    def div(a, c):
        if type(a) is int:
            return a // c
        return a if c == 1 else ("div", a, c)

    def mod(a, c):
        if type(a) is int:
            return a % c
        return 0 if c == 1 else ("mod", a, c)

    def both(a, b):
        if a is True:
            return b
        if b is True:
            return a
        if a is False or b is False:
            return False
        return ("where", a, b, False)

    def pick(c, a, b):
        return a if a == b else ("where", c, a, b)

    def strides(shape):
        result = []
        acc = 1
        for s in reversed(shape):
            result.append(acc)
            acc *= s
        return result[::-1]

    def shapes(sequence, initial=None):
        result = [base if initial is None else tuple(initial)]

        for kind, arg in sequence:
            sh = result[-1]

            if kind in ("reshape", "expand"):
                sh = tuple(arg)

            elif kind == "permute":
                sh = tuple(sh[i] for i in arg)

            elif kind == "shrink":
                sh = tuple(e - b for b, e in arg)

            elif kind == "pad":
                sh = tuple(
                    s + b + a
                    for s, (b, a) in zip(sh, arg)
                )

            elif kind == "stack":
                sh = (len(arg),) + shapes(arg[0], sh)[-1]

            result.append(sh)

        return result

    def lower_ops(sequence, idx):
        shp = shapes(sequence)
        valid = True
        idx = list(idx)

        for k in range(len(sequence) - 1, -1, -1):
            kind, arg = sequence[k]
            ins, outs = shp[k], shp[k + 1]

            if kind == "stack":
                sel, rest = idx[0], idx[1:]

                res = [
                    lower_ops(list(sequence[:k]) + list(branch), rest)
                    for branch in arg
                ]

                flat, v = res[-1]

                for b in range(len(res) - 2, -1, -1):
                    cond = ("cmplt", sel, b + 1)
                    flat = pick(cond, res[b][0], flat)
                    v = pick(cond, res[b][1], v)

                return flat, both(valid, v)

            elif kind == "reshape":
                lin = 0

                for i, s in zip(idx, strides(outs)):
                    lin = add(lin, mul(i, s))

                if k == 0:
                    return lin, valid

                idx = [
                    mod(div(lin, s), n)
                    for s, n in zip(strides(ins), ins)
                ]

            elif kind == "permute":
                new = [None] * len(idx)

                for i, p in enumerate(arg):
                    new[p] = idx[i]

                idx = new

            elif kind == "expand":
                idx = [
                    0 if n == 1 and o != 1 else i
                    for i, n, o in zip(idx, ins, outs)
                ]

            elif kind == "shrink":
                idx = [
                    add(i, b)
                    for i, (b, e) in zip(idx, arg)
                ]

            elif kind == "pad":
                new = []

                for i, (b, a), n in zip(idx, arg, ins):
                    if b > 0:
                        valid = both(
                            valid,
                            ("cmplt", b - 1, i)
                        )

                    if a > 0:
                        valid = both(
                            valid,
                            ("cmplt", i, b + n)
                        )

                    new.append(add(i, -b))

                idx = new

            elif kind == "flip":
                idx = [
                    add(mul(i, -1), ins[ax] - 1)
                    if ax in arg else i
                    for ax, i in enumerate(idx)
                ]

        flat = 0

        for i, s in zip(idx, strides(base)):
            flat = add(flat, mul(i, s))

        return flat, valid

    def simplify(root):
        memo = {}
        bounds_cache = {}

        def is_const(x):
            return type(x) is int

        def interval(x):
            if is_const(x):
                return x, x

            if type(x) is bool:
                return int(x), int(x)

            if not isinstance(x, tuple):
                return -(2**31), 2**31 - 1

            if x in bounds_cache:
                return bounds_cache[x]

            op = x[0]

            if op == "range":
                i = x[1]
                result = (0, out_shape[i] - 1)

            elif op == "add":
                a = interval(x[1])
                b = interval(x[2])
                result = (a[0] + b[0], a[1] + b[1])

            elif op == "mul":
                a = interval(x[1])
                b = interval(x[2])
                values = (
                    a[0] * b[0], a[0] * b[1],
                    a[1] * b[0], a[1] * b[1]
                )
                result = min(values), max(values)

            elif op == "div":
                a = interval(x[1])
                c = x[2]
                if c > 0:
                    result = a[0] // c, a[1] // c
                elif c < 0:
                    result = a[1] // c, a[0] // c
                else:
                    result = (-(2**31), 2**31 - 1)

            elif op == "mod":
                a = interval(x[1])
                c = x[2]
                if c > 0:
                    result = (0, c - 1)
                elif c < 0:
                    result = (c + 1, 0)
                else:
                    result = (-(2**31), 2**31 - 1)

            elif op == "cmplt":
                a = interval(x[1])
                b = interval(x[2])
                if a[1] < b[0]:
                    result = (1, 1)
                elif a[0] >= b[1]:
                    result = (0, 0)
                else:
                    result = (0, 1)

            elif op == "where":
                a = interval(x[2])
                b = interval(x[3])
                result = min(a[0], b[0]), max(a[1], b[1])

            elif op == "max":
                a = interval(x[1])
                b = interval(x[2])
                result = max(a[0], b[0]), max(a[1], b[1])

            else:
                result = (-(2**31), 2**31 - 1)

            bounds_cache[x] = result
            return result

        def affine(x):
            """Return constant and coefficient dictionary."""
            if is_const(x):
                return x, {}

            if isinstance(x, tuple) and x[0] == "add":
                c1, d1 = affine(x[1])
                c2, d2 = affine(x[2])
                d = dict(d1)

                for key, value in d2.items():
                    d[key] = d.get(key, 0) + value

                return c1 + c2, d

            if isinstance(x, tuple) and x[0] == "mul":
                if is_const(x[1]):
                    c, d = affine(x[2])
                    return c * x[1], {
                        key: value * x[1]
                        for key, value in d.items()
                    }

                if is_const(x[2]):
                    c, d = affine(x[1])
                    return c * x[2], {
                        key: value * x[2]
                        for key, value in d.items()
                    }

            return 0, {x: 1}

        def build_affine(c, d):
            result = c

            for atom, coefficient in d.items():
                if coefficient:
                    result = add(
                        result,
                        mul(coefficient, atom)
                    )

            return result

        def divide_affine(x, divisor):
            """Split x into divisor*q + remainder."""
            c, terms = affine(x)

            cq, cr = divmod(c, divisor)
            qterms = {}
            rterms = {}

            for atom, coefficient in terms.items():
                q, r = divmod(coefficient, divisor)

                if q:
                    qterms[atom] = q

                if r:
                    rterms[atom] = r

            qexpr = build_affine(cq, qterms)
            rexpr = build_affine(cr, rterms)

            lo, hi = interval(rexpr)

            if lo >= 0 and hi < divisor:
                return qexpr, rexpr

            return None

        def visit(x):
            if not isinstance(x, tuple):
                return x

            if x in memo:
                return memo[x]

            op = x[0]

            if op == "range":
                return x

            if op == "add":
                a = visit(x[1])
                b = visit(x[2])
                result = build_affine(*affine(add(a, b)))

            elif op == "mul":
                a = visit(x[1])
                b = visit(x[2])
                result = mul(a, b)

            elif op in ("div", "mod"):
                a = visit(x[1])
                c = x[2]

                if c == 1:
                    result = a if op == "div" else 0

                elif c == -1:
                    result = mul(-1, a) if op == "div" else 0

                elif is_const(a):
                    result = a // c if op == "div" else a % c

                elif c > 0:
                    lo, hi = interval(a)

                    if lo >= 0 and hi < c:
                        result = 0 if op == "div" else a

                    else:
                        split = divide_affine(a, c)

                        if split is not None:
                            q, r = split
                            result = q if op == "div" else r

                        elif op == "div" and isinstance(a, tuple) and a[0] == "div":
                            d = a[2]
                            if d > 0 and interval(a[1])[0] >= 0:
                                result = ("div", a[1], d * c)
                            else:
                                result = ("div", a, c)

                        else:
                            result = (op, a, c)

                else:
                    result = (op, a, c)

            elif op == "cmplt":
                a = visit(x[1])
                b = visit(x[2])
                ia, ib = interval(a), interval(b)

                if ia[1] < ib[0]:
                    result = True
                elif ia[0] >= ib[1]:
                    result = False
                else:
                    result = ("cmplt", a, b)

            elif op == "where":
                c = visit(x[1])
                a = visit(x[2])
                b = visit(x[3])

                if c is True:
                    result = a
                elif c is False:
                    result = b
                elif a == b:
                    result = a
                else:
                    result = ("where", c, a, b)

            elif op == "max":
                a = visit(x[1])
                b = visit(x[2])
                ia, ib = interval(a), interval(b)

                if ia[0] >= ib[1]:
                    result = a
                elif ib[0] >= ia[1]:
                    result = b
                elif a == b:
                    result = a
                else:
                    result = ("max", a, b)

            elif op == "xor":
                a = visit(x[1])
                b = visit(x[2])

                if a == b:
                    result = False if type(a) is bool else 0
                else:
                    result = ("xor", a, b)

            elif op == "shl":
                a = visit(x[1])
                b = visit(x[2])
                result = a if b == 0 else ("shl", a, b)

            elif op == "cast":
                a = visit(x[1])
                dtype = x[2]

                if dtype == "i32" and type(a) is bool:
                    result = int(a)
                elif dtype == "bool" and is_const(a):
                    result = bool(a)
                else:
                    result = ("cast", a, dtype)

            else:
                result = tuple(
                    [op] + [
                        visit(v) if isinstance(v, tuple) else v
                        for v in x[1:]
                    ]
                )

            memo[x] = result
            return result

        return visit(root)

    index_expr, valid_expr = lower_ops(
        list(ops),
        [("range", i) for i in range(len(out_shape))]
    )

    return simplify(index_expr), simplify(valid_expr)