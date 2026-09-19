import numpy as np

def svd_2x2(A: np.ndarray) -> tuple:
    """
    Compute the SVD of a 2x2 matrix without using np.linalg.svd.

    Returns U, s, V such that:

        A ≈ U @ np.diag(s) @ V

    Here V is the transpose of the conventional right-singular-vector
    matrix.

    Args:
        A: 2x2 numpy array

    Returns:
        U: 2x2 orthogonal matrix
        s: 1D array of two non-negative singular values
        V: 2x2 matrix (V^T in conventional SVD notation)
    """

    A = np.asarray(A, dtype=float)

    if A.shape != (2, 2):
        raise ValueError("A must be a 2x2 matrix")

    # A^T A is symmetric:
    # [[a, b],
    #  [b, d]]
    ATA = A.T @ A

    a = ATA[0, 0]
    b = ATA[0, 1]
    d = ATA[1, 1]

    # Eigenvalues of A^T A.
    trace = a + d
    diff = a - d
    discriminant = np.sqrt(diff * diff + 4.0 * b * b)

    eig1 = (trace + discriminant) / 2.0
    eig2 = (trace - discriminant) / 2.0

    eig1 = max(eig1, 0.0)
    eig2 = max(eig2, 0.0)

    # Singular values are square roots of eigenvalues.
    s = np.array([
        np.sqrt(eig1),
        np.sqrt(eig2)
    ])

    # Eigenvector for the largest eigenvalue.
    if abs(b) > 1e-14 or abs(eig1 - a) > 1e-14:
        v1 = np.array([b, eig1 - a], dtype=float)

        if np.linalg.norm(v1) < 1e-14:
            v1 = np.array([eig1 - d, b], dtype=float)

        v1 /= np.linalg.norm(v1)
    else:
        # ATA is diagonal / proportional to identity.
        v1 = np.array([1.0, 0.0])

    # The second eigenvector is perpendicular to the first.
    v2 = np.array([-v1[1], v1[0]])

    # V contains the right singular vectors as rows,
    # because the requested reconstruction is U @ diag(s) @ V.
    V = np.vstack([v1, v2])

    # Compute left singular vectors:
    #
    # u_i = A v_i / sigma_i
    U = np.zeros((2, 2))

    if s[0] > 1e-14:
        U[:, 0] = A @ v1 / s[0]

    if s[1] > 1e-14:
        U[:, 1] = A @ v2 / s[1]

    # Complete U if a singular value is zero.
    if s[0] <= 1e-14 and s[1] <= 1e-14:
        U = np.eye(2)

    elif s[0] <= 1e-14:
        # u2 is already defined; choose its perpendicular.
        U[:, 0] = np.array([-U[1, 1], U[0, 1]])

    elif s[1] <= 1e-14:
        # u1 is already defined; choose its perpendicular.
        U[:, 1] = np.array([-U[1, 0], U[0, 0]])

    return U, s, V