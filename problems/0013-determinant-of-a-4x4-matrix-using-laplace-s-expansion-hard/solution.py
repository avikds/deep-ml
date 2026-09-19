def determinant_4x4(matrix: list[list[int | float]]) -> float:
    # Recursively calculate determinant using Laplace expansion.
    def determinant(mat):
        n = len(mat)

        if n == 1:
            return mat[0][0]

        if n == 2:
            return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]

        det = 0

        # Expand along the first row.
        for col in range(n):
            # Construct the minor by removing row 0 and column col.
            minor = [
                [mat[i][j] for j in range(n) if j != col]
                for i in range(1, n)
            ]

            # Cofactor sign: (-1)^(row + column)
            sign = 1 if col % 2 == 0 else -1

            det += sign * mat[0][col] * determinant(minor)

        return det

    return determinant(matrix)