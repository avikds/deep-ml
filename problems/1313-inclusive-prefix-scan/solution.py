#include <cuda_runtime.h>

__global__ void inclusive_scan_kernel(const float* input, float* output, int n) {
    extern __shared__ float smem[];

    float* src = smem;
    float* dst = smem + n;

    int tid = threadIdx.x;

    // Load input into shared memory.
    src[tid] = input[tid];
    __syncthreads();

    // Hillis-Steele inclusive scan with double buffering.
    for (int offset = 1; offset < n; offset <<= 1) {
        float value = src[tid];

        if (tid >= offset) {
            value += src[tid - offset];
        }

        dst[tid] = value;
        __syncthreads();

        float* tmp = src;
        src = dst;
        dst = tmp;

        __syncthreads();
    }

    output[tid] = src[tid];
}

void solve(const float* input, float* output, int n) {
    float* d_input = nullptr;
    float* d_output = nullptr;

    size_t bytes = static_cast<size_t>(n) * sizeof(float);
    size_t shared_bytes = 2 * bytes;

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    cudaMemcpy(d_input, input, bytes, cudaMemcpyHostToDevice);

    inclusive_scan_kernel<<<1, n, shared_bytes>>>(d_input, d_output, n);

    cudaMemcpy(output, d_output, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}