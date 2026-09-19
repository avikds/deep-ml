import numpy as np

def tdnn_forward(sequences, layer_configs):
    """
    Forward pass through a multi-layer Time-Delay Neural Network.
    """
    results = []

    for sequence in sequences:
        x = np.asarray(sequence, dtype=float)
        valid = True

        for config in layer_configs:
            weights = np.asarray(config["weights"], dtype=float)
            bias = np.asarray(config["bias"], dtype=float)
            offsets = config["offsets"]
            activation = config["activation"]

            T = x.shape[0]

            if T == 0:
                valid = False
                break

            min_offset = min(offsets)
            max_offset = max(offsets)

            # Valid center positions t satisfy:
            # 0 <= t + min_offset and t + max_offset < T
            start = max(0, -min_offset)
            end = min(T, T - max_offset)

            if start >= end:
                valid = False
                break

            outputs = []

            for t in range(start, end):
                context = np.concatenate(
                    [x[t + offset] for offset in offsets]
                )

                y = context @ weights + bias

                if activation == "relu":
                    y = np.maximum(y, 0.0)
                elif activation == "none":
                    pass

                outputs.append(y)

            x = np.asarray(outputs, dtype=float)

        if not valid or len(layer_configs) == 0 and x.size == 0:
            results.append([])
        else:
            results.append(np.round(x, 4).tolist())

    return results