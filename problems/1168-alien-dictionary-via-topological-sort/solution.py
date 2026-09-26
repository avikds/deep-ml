def alien_order(words):
    # Every distinct character must appear in the result.
    chars = set(ch for word in words for ch in word)

    # Adjacency set avoids duplicate edges.
    graph = {ch: set() for ch in chars}
    indegree = {ch: 0 for ch in chars}

    # Derive ordering constraints from adjacent words.
    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i + 1]

        # Invalid: longer word appears before its strict prefix.
        if len(w1) > len(w2) and w1.startswith(w2):
            return ""

        for a, b in zip(w1, w2):
            if a != b:
                if b not in graph[a]:
                    graph[a].add(b)
                    indegree[b] += 1
                break

    # Lexicographically smallest available character first.
    available = sorted(ch for ch in chars if indegree[ch] == 0)
    result = []

    while available:
        ch = available.pop(0)
        result.append(ch)

        for nxt in sorted(graph[ch]):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                available.append(nxt)

        available.sort()

    # A cycle means no valid ordering exists.
    if len(result) != len(chars):
        return ""

    return "".join(result)