import numpy as np

def td_lambda_error_bound(P, r, gamma, Phi, d, lambdas):
    """
    Compute and verify TD(lambda) asymptotic error bounds.
    """
    P = np.asarray(P, dtype=float)
    r = np.asarray(r, dtype=float)
    Phi = np.asarray(Phi, dtype=float)
    d = np.asarray(d, dtype=float)

    n = P.shape[0]
    I = np.eye(n)
    D = np.diag(d)

    # True value function.
    V = np.linalg.solve(I - gamma * P, r)

    def d_norm(x):
        return float(np.sqrt(x @ D @ x))

    # Monte Carlo / D-weighted projection.
    G_mc = Phi.T @ D @ Phi
    b_mc = Phi.T @ D @ V

    w_mc = np.linalg.pinv(G_mc) @ b_mc
    V_mc = Phi @ w_mc
    mc_error = d_norm(V - V_mc)

    # Theoretical amplification bound.
    bound = 1.0 / np.sqrt(1.0 - gamma ** 2)

    td_errors = []
    ratios = []
    bound_holds = []

    for lam in lambdas:
        # (I - gamma*lambda*P)^(-1)
        E = np.linalg.inv(I - gamma * lam * P)

        # TD(lambda) fixed-point system.
        A = Phi.T @ D @ E @ (I - gamma * P) @ Phi
        b = Phi.T @ D @ E @ r

        w_td = np.linalg.pinv(A) @ b
        V_td = Phi @ w_td

        error = d_norm(V - V_td)
        td_errors.append(error)

        # Handle exact zero MC error explicitly.
        if mc_error <= 1e-12:
            if error <= 1e-12:
                ratio = 0.0
            else:
                ratio = np.inf
        else:
            ratio = error / mc_error

        ratios.append(ratio)
        bound_holds.append(bool(ratio <= bound + 1e-8))

    return {
        "true_values": np.round(V, 4).tolist(),
        "mc_error": round(mc_error, 4),
        "td_errors": np.round(td_errors, 4).tolist(),
        "ratios": np.round(ratios, 4).tolist(),
        "bound": round(bound, 4),
        "bound_holds": bound_holds
    }