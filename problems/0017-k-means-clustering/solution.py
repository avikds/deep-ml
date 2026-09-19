def k_means_clustering(
    points: list[tuple[float, ...]],
    k: int,
    initial_centroids: list[tuple[float, ...]],
    max_iterations: int
) -> list[tuple[float, ...]]:

    centroids = [tuple(float(x) for x in c) for c in initial_centroids]

    for _ in range(max_iterations):
        # Assign each point to its nearest centroid.
        clusters = [[] for _ in range(k)]

        for point in points:
            distances = [
                sum((point[d] - centroids[i][d]) ** 2
                    for d in range(len(point)))
                for i in range(k)
            ]

            # min() naturally resolves ties by choosing the first centroid.
            closest = min(range(k), key=lambda i: distances[i])
            clusters[closest].append(point)

        # Calculate new centroids.
        new_centroids = []

        for i in range(k):
            if not clusters[i]:
                # Keep the old centroid if no points are assigned.
                new_centroids.append(centroids[i])
            else:
                dimensions = len(points[0])

                new_centroid = tuple(
                    sum(point[d] for point in clusters[i]) / len(clusters[i])
                    for d in range(dimensions)
                )

                new_centroids.append(new_centroid)

        # Stop if centroids no longer change.
        if new_centroids == centroids:
            centroids = new_centroids
            break

        centroids = new_centroids

    # Round only the final result.
    return [
        tuple(round(value, 4) for value in centroid)
        for centroid in centroids
    ]