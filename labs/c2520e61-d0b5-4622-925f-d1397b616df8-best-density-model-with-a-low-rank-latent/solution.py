import numpy as np
import time
from sklearn.cluster import MiniBatchKMeans

def fit(X_train):
    """
    Fit a mixture of 20 factor analysers, each with 10 latent factors.
    Uses EM with Woodbury identities for efficient likelihood evaluation.
    """
    start = time.process_time()
    n, d = X_train.shape
    K = 20
    L = 10
    floor = 0.00035

    # Run and Submit have different fitting budgets.
    budget = 50.0 if n <= 12000 else 155.0
    deadline = start + budget

    X = np.asarray(X_train, dtype=np.float64)
    global_mean = X.mean(axis=0)
    global_var = np.maximum(X.var(axis=0), floor)

    # ------------------------------------------------
    # 1. Initialize mixture components using clustering.
    # ------------------------------------------------
    km = MiniBatchKMeans(
        n_clusters=K,
        batch_size=1024,
        max_iter=100,
        n_init=3,
        random_state=42,
        reassignment_ratio=0.01
    )

    labels = km.fit_predict(X)

    pi = np.zeros(K, dtype=np.float64)
    mu = np.zeros((K, d), dtype=np.float64)
    psi = np.zeros((K, d), dtype=np.float64)
    W = []

    for k in range(K):
        members = X[labels == k]

        if len(members) < L + 2:
            members = X

        nk = len(members)
        center = members.mean(axis=0)
        centered = members - center

        cov = centered.T @ centered / nk

        # Principal directions initialize the factor loadings.
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1][:L]

        vals = np.maximum(vals[order] - floor, 1e-6)
        vecs = vecs[:, order]

        wk = vecs * np.sqrt(vals)[None, :] * 0.90

        residual = np.diag(cov) - np.sum(wk * wk, axis=1)
        psik = np.maximum(residual, floor)

        pi[k] = nk / n
        mu[k] = center
        psi[k] = psik
        W.append(wk)

    pi = np.maximum(pi, 1e-8)
    pi /= pi.sum()

    # ------------------------------------------------
    # 2. EM iterations.
    # ------------------------------------------------
    eye = np.eye(L)
    log2pi = d * np.log(2.0 * np.pi)
    previous_ll = -np.inf
    iteration = 0

    while time.process_time() < deadline:
        iteration += 1

        log_joint = np.empty((n, K), dtype=np.float64)
        posterior = []
        posterior_cov = []

        # E-step: calculate posterior latent means and responsibilities.
        for k in range(K):
            wk = W[k]
            pk = np.maximum(psi[k], floor)
            invpsi = 1.0 / pk

            dw = wk * invpsi[:, None]
            M = eye + wk.T @ dw
            sign, logdet_M = np.linalg.slogdet(M)

            if sign <= 0:
                logdet_M = 0.0

            beta = np.linalg.inv(M)

            xc = X - mu[k]
            T = (xc * invpsi) @ wk
            ez = T @ beta

            quad = np.sum(xc * xc * invpsi, axis=1)
            quad -= np.sum(T * ez, axis=1)
            quad = np.maximum(quad, 0.0)

            logdet = np.sum(np.log(pk)) + logdet_M

            log_joint[:, k] = (
                np.log(max(pi[k], 1e-12))
                - 0.5 * (log2pi + logdet + quad)
            )

            posterior.append(ez)
            posterior_cov.append(beta)

        # Normalize component responsibilities.
        row_max = np.max(log_joint, axis=1, keepdims=True)
        exp_joint = np.exp(log_joint - row_max)
        row_sum = np.maximum(exp_joint.sum(axis=1, keepdims=True), 1e-300)

        resp = exp_joint / row_sum
        ll = float(np.mean(row_max + np.log(row_sum)))

        # M-step.
        new_pi = np.empty(K, dtype=np.float64)
        new_mu = np.empty_like(mu)
        new_psi = np.empty_like(psi)
        new_W = []

        for k in range(K):
            r = resp[:, k]
            nk = float(r.sum())

            if nk < L + 2:
                new_pi[k] = max(nk / n, 1e-8)
                new_mu[k] = mu[k]
                new_psi[k] = psi[k]
                new_W.append(W[k])
                continue

            ez = posterior[k]
            beta = posterior_cov[k]

            # Expected sufficient statistics.
            sum_z = r @ ez
            Szz = nk * beta + ez.T @ (ez * r[:, None])

            Sx = r @ X
            Sxx = r @ (X * X)

            Sxz = (X.T * r) @ ez

            # Joint update of mean and loading matrix.
            augmented = np.empty((L + 1, L + 1))
            augmented[0, 0] = nk
            augmented[0, 1:] = sum_z
            augmented[1:, 0] = sum_z
            augmented[1:, 1:] = Szz

            augmented += np.eye(L + 1) * 1e-8

            rhs = np.column_stack((Sx, Sxz))
            coeff = np.linalg.solve(augmented, rhs.T).T

            muk = coeff[:, 0]
            wk = coeff[:, 1:]

            # Expected residual variance, including latent uncertainty.
            latent_quad = np.einsum(
                "dl,lm,dm->d", wk, Szz, wk
            )

            cross = np.sum(wk * Sxz, axis=1)
            mean_cross = wk @ sum_z

            residual = (
                Sxx
                - 2.0 * muk * Sx
                - 2.0 * cross
                + nk * muk * muk
                + 2.0 * muk * mean_cross
                + latent_quad
            )

            psik = np.maximum(residual / nk, floor)

            new_pi[k] = nk / n
            new_mu[k] = muk
            new_psi[k] = psik
            new_W.append(wk)

        new_pi = np.maximum(new_pi, 1e-10)
        new_pi /= new_pi.sum()

        pi = new_pi
        mu = new_mu
        psi = new_psi
        W = new_W

        # Stop when likelihood improvement becomes negligible.
        if iteration >= 3 and abs(ll - previous_ll) < 1e-6:
            break

        previous_ll = ll

        if iteration >= 30:
            break

    # Ensure all parameters satisfy the output contract.
    psi = np.maximum(psi, floor)
    pi = np.maximum(pi, 1e-12)
    pi /= pi.sum()

    return {
        "pi": pi,
        "mu": mu,
        "W": W,
        "psi": psi
    }