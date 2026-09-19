import numpy as np

def transform_matrix(
    A: list[list[int | float]],
    T: list[list[int | float]],
    S: list[list[int | float]]
) -> list[list[int | float]]:
    A = np.asarray(A, dtype=float)
    T = np.asarray(T, dtype=float)
    S = np.asarray(S, dtype=float)

    # T and S must be square and dimensionally compatible with A.
    if T.ndim != 2 or S.ndim != 2 or A.ndim != 2:
        return -1

    if T.shape[0] != T.shape[1] or S.shape[0] != S.shape[1]:
        return -1

    if T.shape[0] != A.shape[0] or S.shape[0] != A.shape[1]:
        return -1

    # Check invertibility.
    if np.isclose(np.linalg.det(T), 0.0) or np.isclose(np.linalg.det(S), 0.0):
        return -1

    # T^(-1) A S
    transformed_matrix = np.linalg.inv(T) @ A @ S

    return transformed_matrix.tolist()