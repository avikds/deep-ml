#include <cuda_runtime.h>

__global__ void compact_nonzero_kernel(
    const float* input,
    float* output,
    int* count,
    int n
) {
    extern __shared__ int scan[];

    int tid = threadIdx.x;

    // Flag: 1 for nonzero, 0 for zero.
    scan[tid] = (input[tid] != 0.0f) ? 1 : 0;
    __syncthreads();

    // Blelloch exclusive scan.
    for (int offset = 1; offset < n; offset <<= 1) {
        int idx = (tid + 1) * offset * 2 - 1;

        if (idx < n) {
            scan[idx] += scan[idx - offset];
        }
        __syncthreads();
    }

    // Convert inclusive total to exclusive scan.
    if (tid == 0) {
        scan[n - 1] = 0;
    }
    __syncthreads();

    for (int offset = n >> 1; offset >= 1; offset >>= 1) {
        int idx = (tid + 1) * offset * 2 - 1;

        if (idx < n) {
            int left = scan[idx - offset];
            int total = scan[idx];

            scan[idx - offset] = total;
            scan[idx] = total + left;
        }
        __syncthreads();
    }

    // Scatter survivors to their compacted positions.
    if (input[tid] != 0.0f) {
        output[scan[tid]] = input[tid];
    }

    // Total number of survivors.
    if (tid == n - 1) {
        count[0] = scan[tid] + ((input[tid] != 0.0f) ? 1 : 0);
    }
}

void solve(const float* input, float* output, int* count, int n) {
    float* d_input = nullptr;
    float* d_output = nullptr;
    int* d_count = nullptr;

    size_t bytes = static_cast<size_t>(n) * sizeof(float);

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    cudaMalloc(&d_count, sizeof(int));

    cudaMemcpy(d_input, input, bytes, cudaMemcpyHostToDevice);

    compact_nonzero_kernel<<<1, n, bytes / sizeof(float)>>>(
        d_input,
        d_output,
        d_count,
        n
    );

    cudaMemcpy(output, d_output, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_count);
}