import numpy as np

def approximate(matvec, rmatvec, shape, k, budget):
    """
    Randomized block Krylov approximation using at most `budget`
    matrix-vector products (counted per column).
    """
    m, n = shape
    budget = int(budget)
    rank = min(int(k), m, n)

    def mv(X):
        Y = np.asarray(matvec(X), dtype=np.float64)
        if Y.ndim == 1:
            Y = Y[:, None]
        return Y

    def rmv(X):
        Y = np.asarray(rmatvec(X), dtype=np.float64)
        if Y.ndim == 1:
            Y = Y[:, None]
        return Y

    def orth(X, against=None, reference=None):
        """Stable orthogonal basis, optionally against an existing basis."""
        X = np.asarray(X, dtype=np.float64)

        if against is not None and against.shape[1] > 0:
            X = X - against @ (against.T @ X)
            X = X - against @ (against.T @ X)

        if X.size == 0:
            return np.zeros((X.shape[0], 0), dtype=np.float64)

        U, s, _ = np.linalg.svd(X, full_matrices=False)

        if len(s) == 0 or s[0] == 0:
            return np.zeros((X.shape[0], 0), dtype=np.float64)

        threshold = s[0] * 1e-11
        if reference is not None:
            threshold = max(threshold, reference * 1e-13)

        keep = s > threshold
        return U[:, keep]

    # Start with a modest oversampling width. Reserve products for
    # successive Krylov refinements.
    oversample = min(8, max(1, budget // 10))
    width = min(min(m, n), rank + oversample)

    if width < 1:
        raise ValueError("Matrix dimensions must be positive")

    used = 0

    # Choose the smaller side for the initial random subspace.
    if m <= n:
        # Left-subspace algorithm.
        omega = np.random.standard_normal((n, width))
        Y = mv(omega)
        used += width

        Q = orth(Y)
        if Q.shape[1] == 0:
            raise ValueError("Could not construct initial subspace")

        C = rmv(Q)
        used += Q.shape[1]

        reference = np.linalg.norm(Y, ord=2)

        while True:
            # C = A.T @ Q. Its left singular vectors approximate
            # the dominant right singular vectors of A.
            Uc, s, Vhc = np.linalg.svd(C, full_matrices=False)

            if len(s) == 0:
                break

            remaining = budget - used
            p = min(rank, remaining // 2)

            if p < 1:
                break

            Vtop = Uc[:, :p]

            # Generate the next Krylov block.
            Ynew = mv(Vtop)
            used += p

            Qnew = orth(Ynew, against=Q, reference=reference)

            if Qnew.shape[1] == 0:
                break

            Cnew = rmv(Qnew)
            used += Qnew.shape[1]

            Q = np.concatenate((Q, Qnew), axis=1)
            C = np.concatenate((C, Cnew), axis=1)

        # Final projected SVD.
        Uc, s, Vhc = np.linalg.svd(C, full_matrices=False)

        j = min(rank, len(s))
        U = Q @ Vhc.T[:, :j]
        Vt = Uc[:, :j].T

    else:
        # Right-subspace algorithm, useful when n < m.
        omega = np.random.standard_normal((m, width))
        Y = rmv(omega)
        used += width

        P = orth(Y)
        if P.shape[1] == 0:
            raise ValueError("Could not construct initial subspace")

        B = mv(P)
        used += P.shape[1]

        reference = np.linalg.norm(Y, ord=2)

        while True:
            # B = A @ P. Its left singular vectors approximate
            # the dominant left singular vectors of A.
            Ub, s, Vhb = np.linalg.svd(B, full_matrices=False)

            if len(s) == 0:
                break

            remaining = budget - used
            p = min(rank, remaining // 2)

            if p < 1:
                break

            Utop = Ub[:, :p]

            # Generate the next Krylov block.
            Ynew = rmv(Utop)
            used += p

            Pnew = orth(Ynew, against=P, reference=reference)

            if Pnew.shape[1] == 0:
                break

            Bnew = mv(Pnew)
            used += Pnew.shape[1]

            P = np.concatenate((P, Pnew), axis=1)
            B = np.concatenate((B, Bnew), axis=1)

        # Final projected SVD.
        Ub, s, Vhb = np.linalg.svd(B, full_matrices=False)

        j = min(rank, len(s))
        U = Ub[:, :j]
        Vt = Vhb[:j, :] @ P.T

    return (
        np.asarray(U, dtype=np.float64),
        np.asarray(s[:j], dtype=np.float64),
        np.asarray(Vt, dtype=np.float64),
    )