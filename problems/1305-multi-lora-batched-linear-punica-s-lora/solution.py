import numpy as np

def multi_lora_linear(x, W, A, B, lora_ids, scales):
    """
    Batched linear with a per-row LoRA adapter.

    Returns y of shape (batch, out_features).
    """
    x = np.asarray(x)
    W = np.asarray(W)
    A = np.asarray(A)
    B = np.asarray(B)
    lora_ids = np.asarray(lora_ids, dtype=int)
    scales = np.asarray(scales)

    # Shared base linear layer.
    y = x @ W

    # Apply each adapter to the rows that select it.
    for adapter_id in np.unique(lora_ids[lora_ids >= 0]):
        rows = np.flatnonzero(lora_ids == adapter_id)

        # x_rows @ A_id.T -> (n_rows, rank)
        hidden = x[rows] @ A[adapter_id].T

        # hidden @ B_id.T -> (n_rows, out_features)
        y[rows] += scales[adapter_id] * (hidden @ B[adapter_id].T)

    return y