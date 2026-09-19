import numpy as np

def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    size = len(b)
    x = np.zeros(size, dtype=float)

    for _ in range(n):
        x_new = np.zeros(size, dtype=float)

        for i in range(size):
            # Contribution from all elements except the diagonal.
            s = 0.0
            for j in range(size):
                if i != j:
                    s += A[i, j] * x[j]

            x_new[i] = (b[i] - s) / A[i, i]

        # Update only after all components have been calculated.
        x = x_new

    return [round(value, 4) for value in x]