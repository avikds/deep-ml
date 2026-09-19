def min_remove_to_make_valid(s):
    stack = []
    remove = set()

    for i, ch in enumerate(s):
        if ch == '(':
            stack.append(i)
        elif ch == ')':
            if stack:
                stack.pop()
            else:
                remove.add(i)

    # Any '(' left in stack is unmatched.
    remove.update(stack)

    return ''.join(
        ch for i, ch in enumerate(s)
        if i not in remove
    )