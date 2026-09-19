import numpy as np

def off_policy_mc_control(
    episodes: list,
    behavior_policy: dict,
    n_states: int,
    n_actions: int,
    gamma: float = 1.0
) -> list:
    """
    Off-policy Monte Carlo control using weighted importance sampling.
    """
    Q = np.zeros((n_states, n_actions), dtype=float)
    C = np.zeros((n_states, n_actions), dtype=float)

    for episode in episodes:
        G = 0.0
        W = 1.0

        # Process the episode backward
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]

            G = gamma * G + reward

            C[state, action] += W
            Q[state, action] += (
                W / C[state, action]
            ) * (G - Q[state, action])

            # Greedy target-policy action.
            # np.argmax returns the lowest index in case of ties.
            greedy_action = int(np.argmax(Q[state]))

            # Once the behavior action differs from the target action,
            # the importance-sampling ratio becomes zero and processing
            # this episode terminates.
            if action != greedy_action:
                break

            prob = behavior_policy.get((state, action), 0.0)

            if prob <= 0.0:
                break

            W /= prob

            if not np.isfinite(W):
                break

    return np.round(Q, 4).tolist()