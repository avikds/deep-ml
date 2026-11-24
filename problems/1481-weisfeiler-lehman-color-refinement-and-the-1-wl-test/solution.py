from collections import Counter

def wl_refine(adj, colors):
    """Perform one round of Weisfeiler-Lehman refinement."""

    signatures = []

    for i in range(len(adj)):
        signature = (
            colors[i],
            tuple(sorted(colors[j] for j in adj[i]))
        )
        signatures.append(signature)

    unique_signatures = sorted(set(signatures))

    mapping = {
        signature: i
        for i, signature in enumerate(unique_signatures)
    }

    return [mapping[sig] for sig in signatures]


def wl_iterate(adj, colors=None):
    """Refine until the number of distinct colours stops increasing."""

    n = len(adj)

    if colors is None:
        colors = [0] * n
    else:
        colors = list(colors)

    rounds = 0
    num_colors = len(set(colors))

    while True:
        new_colors = wl_refine(adj, colors)
        new_num_colors = len(set(new_colors))

        if new_num_colors <= num_colors:
            return new_colors, rounds

        colors = new_colors
        num_colors = new_num_colors
        rounds += 1


def color_histogram(colors):
    """Return sorted colour-class sizes."""

    return sorted(Counter(colors).values())


def disjoint_union(adj1, adj2):
    """Combine two graphs with the second graph's nodes offset."""

    n1 = len(adj1)

    result = [list(neighbors) for neighbors in adj1]

    for neighbors in adj2:
        result.append([v + n1 for v in neighbors])

    return result


def wl_test(adj1, adj2):
    """Test whether 1-WL distinguishes two graphs."""

    if len(adj1) != len(adj2):
        return False

    # Refine both graphs jointly.
    combined = disjoint_union(adj1, adj2)

    colors, rounds = wl_iterate(combined)

    n1 = len(adj1)

    colors1 = colors[:n1]
    colors2 = colors[n1:]

    # Compare actual colour identities and their frequencies.
    return Counter(colors1) == Counter(colors2)