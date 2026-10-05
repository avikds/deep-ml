import numbers


class UPat:
    CONST = "const"
    VAR = "var"

    def __init__(self, op=None, src=None, name=None):
        self.op = op
        self.src = src
        self.name = name

    def match(self, expr, store):
        """Match expr against this pattern, recording bindings in store."""

        def expr_op(x):
            if isinstance(x, str):
                return UPat.VAR
            if isinstance(x, numbers.Number):
                return UPat.CONST
            if isinstance(x, tuple) and len(x) > 0:
                return x[0]
            return None

        def op_matches(x):
            if self.op is None:
                return True

            actual = expr_op(x)

            if isinstance(self.op, (set, frozenset)):
                return actual in self.op

            return actual == self.op

        # Check the operator/type constraint.
        if not op_matches(expr):
            return False

        # If child patterns are specified, the expression must be a tuple
        # with exactly the same number of children.
        if self.src is not None:
            if not isinstance(expr, tuple):
                return False
            children = expr[1:]
            if len(children) != len(self.src):
                return False
        else:
            children = None

        # Work on a copy so a failed match does not leak bindings.
        local = dict(store)

        # Bind the whole matched expression if requested.
        if self.name is not None:
            if self.name in local:
                if local[self.name] != expr:
                    return False
            else:
                local[self.name] = expr

        # Match children recursively.
        if self.src is not None:
            for pat, child in zip(self.src, children):
                if not pat.match(child, local):
                    return False

        store.clear()
        store.update(local)
        return True


def rewrite(expr, rules):
    """Bottom-up rewrite to a fixed point."""
    memo = {}

    def _rewrite(node):
        if node in memo:
            return memo[node]

        total = 0
        current = node

        # Rewrite children first.
        if isinstance(current, tuple):
            op = current[0]
            children = current[1:]
            new_children = []

            for child in children:
                new_child, n = _rewrite(child)
                new_children.append(new_child)
                total += n

            current = (op, *new_children)

        # Keep applying rules until none applies.
        while True:
            changed = False

            for pat, fn in rules:
                bindings = {}

                if not pat.match(current, bindings):
                    continue

                replacement = fn(**bindings)

                # None means the rule declines.
                if replacement is None:
                    continue

                # Rewriting to the identical expression does not count
                # as an application, otherwise this could loop forever.
                if replacement == current:
                    continue

                # The replacement itself must be rewritten first.
                rewritten, n = _rewrite(replacement)
                total += 1 + n
                current = rewritten
                changed = True
                break

            if not changed:
                break

        memo[node] = (current, total)
        return memo[node]

    return _rewrite(expr)