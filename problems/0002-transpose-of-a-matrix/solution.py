def transpose_matrix(a: list[list[int | float]]) -> list[list[int | float]]:
    """
    Transpose a 2D matrix by swapping rows and columns.
    """
    if not a:
        return []

    return [list(row) for row in zip(*a)]