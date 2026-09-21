def distributed_stats(nodes, lo, hi):
    # Compact global summary: frequency of each value in [lo, hi].
    counts = [0] * (hi - lo + 1)

    total = 0
    for node in nodes:
        local_counts = [0] * len(counts)

        for value in node:
            local_counts[value - lo] += 1
            total += 1

        # Simulate sending the compact per-node summary to the coordinator.
        for i, count in enumerate(local_counts):
            counts[i] += count

    # Global mode: smallest value wins ties.
    max_count = max(counts)
    mode = lo + counts.index(max_count)

    # Find the k-th value in sorted order using cumulative frequencies.
    def kth(k):
        cumulative = 0
        for i, count in enumerate(counts):
            cumulative += count
            if cumulative > k:
                return lo + i
        return hi

    if total % 2 == 1:
        median = float(kth(total // 2))
    else:
        left = kth(total // 2 - 1)
        right = kth(total // 2)
        median = (left + right) / 2.0

    return int(mode), float(median)