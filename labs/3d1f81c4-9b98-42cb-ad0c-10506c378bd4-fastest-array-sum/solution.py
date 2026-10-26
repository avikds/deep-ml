#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define MAX_BLOCKS 1024
#define NUM_WARPS (BLOCK_SIZE / 32)

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__global__ void reduce_stage1(
    const float* __restrict__ x,
    int n,
    float* __restrict__ partials
) {
    __shared__ float warp_sums[NUM_WARPS];

    const int tid = threadIdx.x;
    const int global_tid = blockIdx.x * BLOCK_SIZE + tid;
    const int total_threads = gridDim.x * BLOCK_SIZE;

    const int n4 = n / 4;
    const float4* x4 = reinterpret_cast<const float4*>(x);

    float s0 = 0.0f;
    float s1 = 0.0f;
    float s2 = 0.0f;
    float s3 = 0.0f;

    // Coalesced vector loads, with multiple independent accumulators.
    for (int i = global_tid; i < n4; i += total_threads) {
        const float4 v = x4[i];
        s0 += v.x;
        s1 += v.y;
        s2 += v.z;
        s3 += v.w;
    }

    float sum = (s0 + s1) + (s2 + s3);

    // Handle any elements remaining after the float4 loads.
    if (blockIdx.x == 0) {
        for (int i = n4 * 4 + tid; i < n; i += BLOCK_SIZE) {
            sum += x[i];
        }
    }

    // Reduce within each warp.
    sum = warp_sum(sum);

    if ((tid & 31) == 0) {
        warp_sums[tid >> 5] = sum;
    }
    __syncthreads();

    // The first warp reduces the warp totals.
    if (tid < 32) {
        sum = (tid < NUM_WARPS) ? warp_sums[tid] : 0.0f;
        sum = warp_sum(sum);

        if (tid == 0) {
            partials[blockIdx.x] = sum;
        }
    }
}

__global__ void reduce_stage2(
    const float* __restrict__ partials,
    int count,
    float* __restrict__ out
) {
    __shared__ float warp_sums[NUM_WARPS];

    const int tid = threadIdx.x;
    float sum = 0.0f;

    for (int i = tid; i < count; i += BLOCK_SIZE) {
        sum += partials[i];
    }

    sum = warp_sum(sum);

    if ((tid & 31) == 0) {
        warp_sums[tid >> 5] = sum;
    }
    __syncthreads();

    if (tid < 32) {
        sum = (tid < NUM_WARPS) ? warp_sums[tid] : 0.0f;
        sum = warp_sum(sum);

        if (tid == 0) {
            out[0] = sum;
        }
    }
}

extern "C" void array_sum(const float* x, int n, float* out) {
    if (n <= 0) {
        cudaMemset(out, 0, sizeof(float));
        return;
    }

    static float* partials = nullptr;
    static int partial_capacity = 0;

    const int n4 = n / 4;
    int blocks = (n4 + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if (blocks < 1) blocks = 1;
    if (blocks > MAX_BLOCKS) blocks = MAX_BLOCKS;

    // Reuse scratch allocation across invocations.
    if (partial_capacity < blocks) {
        if (partials != nullptr) {
            cudaFree(partials);
            partials = nullptr;
        }

        cudaMalloc(&partials, static_cast<size_t>(blocks) * sizeof(float));
        partial_capacity = blocks;
    }

    reduce_stage1<<<blocks, BLOCK_SIZE>>>(x, n, partials);
    reduce_stage2<<<1, BLOCK_SIZE>>>(partials, blocks, out);
}