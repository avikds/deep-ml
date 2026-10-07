def dp_drills(weights, values, capacity, m, n, s1, s2):
    # 0/1 Knapsack
    dp = [0] * (capacity + 1)

    for w, v in zip(weights, values):
        for c in range(capacity, w - 1, -1):
            dp[c] = max(dp[c], dp[c - w] + v)

    knapsack = dp[capacity]

    # Grid paths: number of paths from (0,0) to (m-1,n-1).
    grid = [[0] * n for _ in range(m)]
    grid[0][0] = 1

    for i in range(m):
        for j in range(n):
            if i == 0 and j == 0:
                continue
            top = grid[i - 1][j] if i > 0 else 0
            left = grid[i][j - 1] if j > 0 else 0
            grid[i][j] = top + left

    grid_paths = grid[m - 1][n - 1]

    # Longest Common Subsequence.
    l1, l2 = len(s1), len(s2)
    lcs_dp = [[0] * (l2 + 1) for _ in range(l1 + 1)]

    for i in range(1, l1 + 1):
        for j in range(1, l2 + 1):
            if s1[i - 1] == s2[j - 1]:
                lcs_dp[i][j] = lcs_dp[i - 1][j - 1] + 1
            else:
                lcs_dp[i][j] = max(
                    lcs_dp[i - 1][j],
                    lcs_dp[i][j - 1]
                )

    lcs = lcs_dp[l1][l2]

    complexity = {
        "knapsack": "O(n*W) time, O(n*W) space",
        "grid_paths": "O(m*n) time, O(m*n) space",
        "lcs": "O(m*n) time, O(m*n) space",
    }

    return {
        "knapsack": knapsack,
        "grid_paths": grid_paths,
        "lcs": lcs,
        "complexity": complexity,
    }