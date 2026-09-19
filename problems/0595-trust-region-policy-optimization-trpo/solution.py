import numpy as np

def trpo_step(
    theta: np.ndarray,
    states: np.ndarray,
    actions: np.ndarray,
    advantages: np.ndarray,
    num_states: int,
    num_actions: int,
    delta: float,
    cg_iters: int = 10,
    line_search_steps: int = 10,
    line_search_decay: float = 0.5
) -> np.ndarray:
    """
    Perform a single TRPO policy update for a tabular softmax policy.
    """
    theta = np.asarray(theta, dtype=float)
    states = np.asarray(states, dtype=int)
    actions = np.asarray(actions, dtype=int)
    advantages = np.asarray(advantages, dtype=float)

    n_params = num_states * num_actions
    old_theta = theta.copy()

    def softmax(logits):
        logits = logits - np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits)
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    old_policy = softmax(
        old_theta.reshape(num_states, num_actions)
    )

    # State visitation frequencies.
    state_freq = np.bincount(
        states, minlength=num_states
    ).astype(float)

    if len(states) > 0:
        state_freq /= len(states)

    # ----- Surrogate objective and its gradient -----
    grad = np.zeros(n_params, dtype=float)

    for s, a, adv in zip(states, actions, advantages):
        p = old_policy[s]
        score = -p
        score[a] += 1.0

        start = s * num_actions
        grad[start:start + num_actions] += adv * score

    if len(states) > 0:
        grad /= len(states)

    # ----- Fisher information matrix -----
    # For each state:
    # F_s = diag(p) - p p^T
    # weighted by the state's visitation frequency.
    F = np.zeros((n_params, n_params), dtype=float)

    for s in range(num_states):
        p = old_policy[s]
        block = np.diag(p) - np.outer(p, p)

        start = s * num_actions
        F[
            start:start + num_actions,
            start:start + num_actions
        ] = state_freq[s] * block

    # Small damping for numerical stability.
    damping = 1e-8
    F += damping * np.eye(n_params)

    # ----- Conjugate gradient: F x = grad -----
    x = np.zeros_like(grad)
    r = grad.copy()
    p = r.copy()
    rs_old = np.dot(r, r)

    if rs_old > 0.0:
        for _ in range(cg_iters):
            Ap = F @ p
            denom = np.dot(p, Ap)

            if abs(denom) < 1e-15:
                break

            alpha = rs_old / denom
            x += alpha * p
            r -= alpha * Ap

            rs_new = np.dot(r, r)

            if np.sqrt(rs_new) < 1e-10:
                break

            p = r + (rs_new / rs_old) * p
            rs_old = rs_new

    # No useful gradient direction.
    if np.linalg.norm(x) < 1e-12:
        return old_theta

    # ----- Maximum step satisfying quadratic KL constraint -----
    xFx = np.dot(x, F @ x)

    if xFx <= 0.0:
        return old_theta

    step_scale = np.sqrt(2.0 * delta / xFx)
    full_step = step_scale * x

    old_flat_policy = old_policy.reshape(-1)

    def evaluate(new_theta):
        new_policy = softmax(
            new_theta.reshape(num_states, num_actions)
        )

        # Exact surrogate objective:
        # E[rho_t * A_t]
        new_probs_taken = new_policy[states, actions]
        old_probs_taken = old_flat_policy[
            states * num_actions + actions
        ]

        ratios = new_probs_taken / np.clip(
            old_probs_taken, 1e-12, None
        )
        surrogate = np.mean(ratios * advantages)

        # State-visitation weighted KL(old || new).
        kl = 0.0
        for s in range(num_states):
            if state_freq[s] == 0.0:
                continue

            p_old = old_policy[s]
            p_new = new_policy[s]

            kl_s = np.sum(
                p_old
                * (
                    np.log(np.clip(p_old, 1e-12, None))
                    - np.log(np.clip(p_new, 1e-12, None))
                )
            )
            kl += state_freq[s] * kl_s

        return surrogate, kl

    old_surrogate = 0.0
    if len(advantages) > 0:
        old_surrogate = float(np.mean(advantages))

    # ----- Backtracking line search -----
    for i in range(line_search_steps):
        fraction = line_search_decay ** i
        candidate = old_theta + fraction * full_step

        candidate_surrogate, candidate_kl = evaluate(candidate)

        if (
            np.isfinite(candidate_surrogate)
            and np.isfinite(candidate_kl)
            and candidate_kl <= delta
            and candidate_surrogate > old_surrogate
        ):
            return candidate

    return old_theta