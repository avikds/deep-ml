import numpy as np

def qr_decomposition(
    A: list[list[float]]
) -> tuple[list[list[float]], list[list[float]]]:
    """
    Perform QR decomposition using the Gram-Schmidt process.
    """
    A = np.asarray(A, dtype=float)
    m, n = A.shape

    Q = np.zeros((m, n), dtype=float)
    R = np.zeros((n, n), dtype=float)

    for j in range(n):
        v = A[:, j].copy()

        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v -= R[i, j] * Q[:, i]

        R[j, j] = np.linalg.norm(v)

        if R[j, j] < 1e-12:
            raise ValueError("Matrix columns are linearly dependent")

        Q[:, j] = v / R[j, j]

    return Q.tolist(), R.tolist()