def attention_memory(T: int, H: int, W: int, num_heads: int, bytes_per_element: int) -> list:
    """
    Compute memory (in bytes) of full and space-time factorized attention scores.
    """
    S = H * W
    N = T * S

    # Full attention: one (N x N) matrix per head
    full_bytes = num_heads * N * N * bytes_per_element

    # Factorized attention:
    # Spatial: T matrices of size (S x S) per head
    # Temporal: S matrices of size (T x T) per head
    factorized_bytes = (
        num_heads
        * (T * S * S + S * T * T)
        * bytes_per_element
    )

    return [full_bytes, factorized_bytes]