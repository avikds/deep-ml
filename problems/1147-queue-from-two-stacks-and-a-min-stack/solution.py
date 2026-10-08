def process_operations(operations):
    # Queue implemented with two stacks.
    inbox = []
    outbox = []

    # Min-stack: each entry stores (value, minimum_so_far).
    min_stack = []

    outputs = []

    for op in operations:
        name = op[0]

        if name == "enqueue":
            inbox.append(op[1])

        elif name == "dequeue":
            if not outbox:
                while inbox:
                    outbox.append(inbox.pop())
            outputs.append(outbox.pop())

        elif name == "mpush":
            v = op[1]
            current_min = v if not min_stack else min(v, min_stack[-1][1])
            min_stack.append((v, current_min))

        elif name == "mpop":
            outputs.append(min_stack.pop()[0])

        elif name == "mtop":
            outputs.append(min_stack[-1][0])

        elif name == "mmin":
            outputs.append(min_stack[-1][1])

    return outputs