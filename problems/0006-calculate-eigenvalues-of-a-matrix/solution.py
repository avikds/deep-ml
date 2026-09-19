def calculate_eigenvalues(matrix: list[list[float | int]]) -> list[float]:
    a, b = matrix[0]
    c, d = matrix[1]

    trace = a + d
    determinant = a * d - b * c

    discriminant = trace**2 - 4 * determinant

    eigenvalue_1 = (trace + discriminant**0.5) / 2
    eigenvalue_2 = (trace - discriminant**0.5) / 2

    eigenvalues = [eigenvalue_1, eigenvalue_2]
    eigenvalues.sort(reverse=True)

    return eigenvalues