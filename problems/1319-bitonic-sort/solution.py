#include <cuda_runtime.h>

__global__ void bitonic_sort_kernel(const float* input, float* output, int n) {
    extern __shared__ float shared[];

    int tid = threadIdx.x;

    shared[tid] = input[tid];
    __syncthreads();

    // Bitonic sorting network.
    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int partner = tid ^ j;

            // Process each pair exactly once.
            if (partner > tid) {
                float a = shared[tid];
                float b = shared[partner];

                // Alternating ascending and descending sequences.
                bool ascending = ((tid & k) == 0);

                if ((ascending && a > b) || (!ascending && a < b)) {
                    shared[tid] = b;
                    shared[partner] = a;
                }
            }

            __syncthreads();
        }
    }

    output[tid] = shared[tid];
}

void solve(const float* input, float* output, int n) {
    size_t bytes = static_cast<size_t>(n) * sizeof(float);

    float* d_input = nullptr;
    float* d_output = nullptr;

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    cudaMemcpy(d_input, input, bytes, cudaMemcpyHostToDevice);

    bitonic_sort_kernel<<<1, n, bytes>>>(d_input, d_output, n);

    cudaMemcpy(output, d_output, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}