import numpy as np


def n_step_tree_backup(
    Q,
    states,
    actions,
    rewards,
    target_policy,
    terminal_states,
    n,
    alpha,
    gamma
):
    """
    Implement the n-step tree backup algorithm for a single episode.
    """

    Q = np.asarray(Q, dtype=float).copy()
    target_policy = np.asarray(target_policy, dtype=float)

    T = len(actions)

    if len(states) != T + 1:
        raise ValueError("states must have length len(actions) + 1")

    if len(rewards) != T:
        raise ValueError("rewards must have length len(actions)")

    if n <= 0:
        raise ValueError("n must be positive")

    terminal_set = set(terminal_states)

    for t in range(T):
        # Effective horizon.
        tau = min(t + n, T)

        # ---------------------------------------------------------
        # Leaf value at S_tau.
        # ---------------------------------------------------------
        leaf_state = states[tau]

        if leaf_state in terminal_set:
            G = 0.0
        else:
            # Expected Q under the target policy.
            G = float(
                np.sum(
                    target_policy[leaf_state] * Q[leaf_state]
                )
            )

        # ---------------------------------------------------------
        # Back up through intermediate states:
        # k = tau-1, ..., t+1
        #
        # G_k =
        #   sum_{a != A_k} pi(a|S_k) Q(S_k,a)
        #   + pi(A_k|S_k) [R_k + gamma * G_{k+1}]
        # ---------------------------------------------------------
        for k in range(tau - 1, t, -1):
            state = states[k]
            action = actions[k]
            reward = rewards[k]

            probs = target_policy[state]

            # Contribution from actions other than the action
            # actually taken in the trajectory.
            branch_value = float(
                np.dot(probs, Q[state])
            )

            # Replace the taken action's ordinary Q contribution
            # with its recursive tree-backup continuation.
            branch_value -= (
                probs[action] * Q[state, action]
            )

            branch_value += (
                probs[action]
                * (reward + gamma * G)
            )

            G = branch_value

        # ---------------------------------------------------------
        # Add the immediate reward for the outermost action.
        # ---------------------------------------------------------
        target = rewards[t] + gamma * G

        # ---------------------------------------------------------
        # Update immediately so later timesteps see the new Q.
        # ---------------------------------------------------------
        state = states[t]
        action = actions[t]

        Q[state, action] += alpha * (
            target - Q[state, action]
        )

    return np.round(Q, 4)