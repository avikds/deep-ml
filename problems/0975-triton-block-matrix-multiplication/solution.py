import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Program IDs identify the output tile.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Float32 accumulation regardless of input dtype.
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over K in BLOCK_K-sized chunks.
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)

        a_ptrs = (
            a_ptr
            + offs_m[:, None] * stride_am
            + offs_k[None, :] * stride_ak
        )
        b_ptrs = (
            b_ptr
            + offs_k[:, None] * stride_bk
            + offs_n[None, :] * stride_bn
        )

        a_mask = (
            (offs_m[:, None] < M) &
            (offs_k[None, :] < K)
        )
        b_mask = (
            (offs_k[:, None] < K) &
            (offs_n[None, :] < N)
        )

        tile_a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        tile_b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(tile_a, tile_b)

    # Cast accumulator back to the output/input dtype.
    c = acc.to(c_ptr.dtype.element_ty)

    c_ptrs = (
        c_ptr
        + offs_m[:, None] * stride_cm
        + offs_n[None, :] * stride_cn
    )

    c_mask = (
        (offs_m[:, None] < M) &
        (offs_n[None, :] < N)
    )

    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("a and b must be 2-D tensors")

    M, K = a.shape
    K_b, N = b.shape

    if K != K_b:
        raise ValueError("incompatible matrix dimensions")

    if a.dtype != b.dtype:
        raise ValueError("a and b must have the same dtype")

    if not a.is_cuda or not b.is_cuda:
        raise ValueError("a and b must be CUDA tensors")

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 32

    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N),
    )

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return c