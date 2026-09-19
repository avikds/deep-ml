def gambler_value_iteration(ph, theta=1e-9):
    """
    Computes the optimal value function and policy for the Gambler's Problem.
    """
    if not 0.0 <= ph <= 1.0:
        raise ValueError("ph must be between 0 and 1")

    V = [0.0] * 101
    V[100] = 1.0
    policy = [0] * 101

    while True:
        delta = 0.0

        for s in range(1, 100):
            old_value = V[s]
            best_value = -1.0
            best_action = 0

            for a in range(1, min(s, 100 - s) + 1):
                win_state = s + a
                lose_state = s - a

                # If the win reaches 100, receive reward 1 and
                # do not add V[100] again.
                if win_state == 100:
                    win_value = 1.0
                else:
                    win_value = V[win_state]

                value = (
                    ph * win_value
                    + (1.0 - ph) * V[lose_state]
                )

                if value > best_value:
                    best_value = value
                    best_action = a

            V[s] = best_value
            policy[s] = best_action

            delta = max(delta, abs(V[s] - old_value))

        if delta < theta:
            break

    return V, policy