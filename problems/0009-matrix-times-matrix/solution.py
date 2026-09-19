def matrixmul(a: list[list[int | float]],
              b: list[list[int | float]]) -> list[list[int | float]]:
    # Check that matrices are non-empty and rectangular
    if not a or not b or not a[0] or not b[0]:
        return -1

    # A is m x n, B is p x q.
    # Multiplication requires n == p.
    if len(a[0]) != len(b):
        return -1

    # Compute C = A @ B
    rows = len(a)
    cols = len(b[0])

    c = [[0 for _ in range(cols)] for _ in range(rows)]

    for i in range(rows):
        for j in range(cols):
            for k in range(len(b)):
                c[i][j] += a[i][k] * b[k][j]

    return c