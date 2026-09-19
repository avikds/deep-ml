import numpy as np

def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    """
    Compute an approximate SVD of a 2x2 matrix using one Jacobi rotation.

    Args:
        A: A 2x2 numpy array

    Returns:
        Tuple (U, S, Vt) where A ≈ U @ diag(S) @ Vt
    """
    # Form the symmetric matrix A^T A.
    B = A.T @ A

    # Elements of B.
    a = B[0, 0]
    b = B[0, 1]
    d = B[1, 1]

    # Compute the Jacobi rotation angle that diagonalizes B.
    if abs(b) < 1e-15:
        c = 1.0
        s = 0.0
    else:
        theta = 0.5 * np.arctan2(2.0 * b, a - d)
        c = np.cos(theta)
        s = np.sin(theta)

    # Jacobi rotation.
    V = np.array([
        [c, -s],
        [s,  c]
    ], dtype=float)

    # Eigenvalues of A^T A after the rotation.
    D = V.T @ B @ V
    eigenvalues = np.diag(D)

    # Singular values are square roots of the eigenvalues.
    S = np.sqrt(np.maximum(eigenvalues, 0.0))

    # Sort singular values from largest to smallest.
    order = np.argsort(S)[::-1]
    S = S[order]
    V = V[:, order]

    # Compute U = A V / S.
    U = np.zeros((2, 2), dtype=float)

    tol = 1e-12

    for i in range(2):
        if S[i] > tol:
            U[:, i] = (A @ V[:, i]) / S[i]

    # If a singular value is zero, construct an orthogonal
    # unit vector for the corresponding column of U.
    if S[0] <= tol:
        U[:, 0] = np.array([1.0, 0.0])

    if S[1] <= tol:
        U[:, 1] = np.array([-U[1, 0], U[0, 0]])

    # Normalize U to protect against small floating-point errors.
    for i in range(2):
        norm = np.linalg.norm(U[:, i])
        if norm > tol:
            U[:, i] /= norm

    Vt = V.T

    return U, S, Vt