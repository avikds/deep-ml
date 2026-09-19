import numpy as np

def ring_attention_simulate(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    num_devices: int
) -> tuple:
    Q = np.asarray(Q, dtype=float)
    K = np.asarray(K, dtype=float)
    V = np.asarray(V, dtype=float)

    seq_len, d = Q.shape

    if num_devices <= 0 or seq_len % num_devices != 0:
        raise ValueError("seq_len must be divisible by num_devices")

    if K.shape != Q.shape or V.shape != Q.shape:
        raise ValueError("Q, K, and V must have the same shape")

    chunk = seq_len // num_devices
    scale = 1.0 / np.sqrt(d)

    Q_chunks = [
        Q[i * chunk:(i + 1) * chunk]
        for i in range(num_devices)
    ]
    K_chunks = [
        K[i * chunk:(i + 1) * chunk]
        for i in range(num_devices)
    ]
    V_chunks = [
        V[i * chunk:(i + 1) * chunk]
        for i in range(num_devices)
    ]

    # Online-softmax accumulators for each device.
    m = [np.full(chunk, -np.inf) for _ in range(num_devices)]
    l = [np.zeros(chunk) for _ in range(num_devices)]
    acc = [np.zeros((chunk, d)) for _ in range(num_devices)]

    comm_schedule = []

    for step in range(num_devices):
        step_sources = []

        for device in range(num_devices):
            # Ring rotates in the expected direction:
            # device 0: 0, n-1, n-2, ...
            source = (device - step) % num_devices
            step_sources.append(source)

            q = Q_chunks[device]
            k = K_chunks[source]
            v = V_chunks[source]

            scores = (q @ k.T) * scale

            block_m = np.max(scores, axis=1)
            p = np.exp(scores - block_m[:, None])
            block_l = np.sum(p, axis=1)

            new_m = np.maximum(m[device], block_m)

            old_scale = np.exp(m[device] - new_m)
            block_scale = np.exp(block_m - new_m)

            new_l = (
                old_scale * l[device]
                + block_scale * block_l
            )

            acc[device] = (
                old_scale[:, None] * acc[device]
                + block_scale[:, None] * (p @ v)
            )

            m[device] = new_m
            l[device] = new_l

        comm_schedule.append(step_sources)

    output = np.zeros((seq_len, d), dtype=float)

    for device in range(num_devices):
        start = device * chunk
        end = start + chunk

        output[start:end] = (
            acc[device] /
            np.maximum(l[device][:, None], 1e-12)
        )

    return np.round(output, 4), comm_schedule