def emphatic_td(
    trajectory: list,
    gamma: float,
    lam: float,
    alpha: float,
    target_policy: dict,
    behavior_policy: dict,
    interest: dict,
    initial_values: dict
) -> dict:
    """
    Emphatic TD(lambda) for off-policy tabular policy evaluation.
    
    Args:
        trajectory: list of (state, action, reward, next_state)
        gamma: discount factor
        lam: lambda for eligibility traces
        alpha: step size
        target_policy: dict (state, action) -> target probability
        behavior_policy: dict (state, action) -> behavior probability
        interest: dict state -> interest value (default 0 for missing states)
        initial_values: dict state -> initial value estimate
    
    Returns:
        Dict mapping state -> rounded value estimate
    """
    V = {state: float(value) for state, value in initial_values.items()}
    e = {state: 0.0 for state in V}

    F = 0.0
    rho_prev = 0.0

    for state, action, reward, next_state in trajectory:
        # Importance-sampling ratio.
        target_prob = target_policy.get((state, action), 0.0)
        behavior_prob = behavior_policy.get((state, action), 0.0)

        if behavior_prob == 0.0:
            rho = 0.0
        else:
            rho = target_prob / behavior_prob

        # Follow-on trace.
        interest_t = interest.get(state, 0.0)
        F = gamma * rho_prev * F + interest_t

        # Emphasis.
        M = lam * interest_t + (1.0 - lam) * F

        # TD error.
        current_value = V.get(state, 0.0)

        if next_state is None:
            next_value = 0.0
        else:
            next_value = V.get(next_state, 0.0)

        delta = reward + gamma * next_value - current_value

        # Emphatic eligibility trace.
        for s in e:
            e[s] *= gamma * lam * rho

        if state not in e:
            e[state] = 0.0

        e[state] += M * rho

        # Value update.
        for s in e:
            V[s] += alpha * delta * e[s]

        rho_prev = rho

    return {
        state: round(float(V[state]), 4)
        for state in sorted(initial_values.keys())
    }