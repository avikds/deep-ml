import numpy as np


def gtd2_prediction(
    features: np.ndarray,
    transitions: list,
    gamma: float,
    alpha_w: float,
    alpha_v: float,
    w_init: np.ndarray,
    v_init: np.ndarray
) -> tuple:
    """
    Perform GTD2 (Gradient TD with second-order correction) for policy evaluation.
    """

    features = np.asarray(features, dtype=float)

    # Work on copies so the caller's arrays are not modified.
    w = np.asarray(w_init, dtype=float).copy()
    v = np.asarray(v_init, dtype=float).copy()

    for s, reward, s_next, done in transitions:
        s = int(s)
        s_next = int(s_next)
        reward = float(reward)

        # Current feature vector.
        x = features[s]

        # Terminal transitions use zero next-state features.
        if done:
            x_next = np.zeros_like(x)
        else:
            x_next = features[s_next]

        # ---------------------------------------------------------
        # TD error:
        # delta = r + gamma * w^T x_next - w^T x
        # ---------------------------------------------------------
        delta = (
            reward
            + gamma * np.dot(w, x_next)
            - np.dot(w, x)
        )

        # ---------------------------------------------------------
        # GTD2 primary update:
        #
        # w += alpha_w * (x - gamma*x_next) * (x^T v)
        # ---------------------------------------------------------
        correction = np.dot(x, v)

        w += alpha_w * (x - gamma * x_next) * correction

        # ---------------------------------------------------------
        # GTD2 secondary update:
        #
        # v += alpha_v * (delta - x^T v) * x
        # ---------------------------------------------------------
        v += alpha_v * (delta - correction) * x

    return (
        [round(float(val), 4) for val in w],
        [round(float(val), 4) for val in v]
    )