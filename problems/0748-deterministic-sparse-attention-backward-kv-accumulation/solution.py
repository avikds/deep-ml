import numpy as np

def deterministic_sparse_attention_backward(
    contributions: list,
    num_sms: int,
    kv_len: int,
    dim: int
) -> list:
    """
    Deterministically accumulate sparse attention KV gradient contributions.
    """

    # One private accumulation buffer per SM.
    per_sm = np.zeros((num_sms, kv_len, dim), dtype=np.float64)

    # Accumulate within each SM in input order.
    for contribution in contributions:
        sm_id = int(contribution["sm_id"])
        kv_idx = int(contribution["kv_idx"])
        grad = np.asarray(contribution["grad"], dtype=np.float64)

        per_sm[sm_id, kv_idx] += grad

    # Deterministic reduction: SMs are processed in ascending order.
    result = np.zeros((kv_len, dim), dtype=np.float64)

    for sm_id in range(num_sms):
        result += per_sm[sm_id]

    return result.tolist()