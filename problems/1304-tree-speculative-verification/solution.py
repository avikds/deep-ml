def tree_speculative_verify(draft_tokens, parents, target_next):
    """
    Walk a draft tree and return the accepted token id path (root excluded).
    """
    n = len(draft_tokens)

    # Build children in increasing node-index order.
    children = [[] for _ in range(n)]
    root = -1

    for i, p in enumerate(parents):
        if p == -1:
            root = i
        else:
            children[p].append(i)

    accepted = []
    current = root

    while current != -1:
        target = target_next[current]
        next_node = -1

        # Children are already in increasing index order.
        for child in children[current]:
            if draft_tokens[child] == target:
                next_node = child
                break

        if next_node == -1:
            break

        accepted.append(draft_tokens[next_node])
        current = next_node

    return accepted