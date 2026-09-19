def matrix_dot_vector(a: list[list[int | float]], b: list[int | float]) -> list[int | float]:
    # Check that every row has the same number of columns as the vector.
    if not a:
        return []

    if any(len(row) != len(b) for row in a):
        return -1

    # Compute the dot product of each row with the vector.
    return [sum(x * y for x, y in zip(row, b)) for row in a]