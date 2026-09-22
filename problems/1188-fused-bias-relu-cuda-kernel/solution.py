#include <cuda_runtime.h>

__global__ void kernel(
    const float* input,
    const float* bias,
    float* output,
    int rows,
    int cols
) {
    // Flatten the 2D matrix into a 1D array.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;

    if (idx < total) {
        // Column index determines which bias value is used.
        int col = idx % cols;

        // Fuse bias addition and ReLU in a single kernel pass.
        float value = input[idx] + bias[col];
        output[idx] = (value > 0.0f) ? value : 0.0f;
    }
}

void solve(
    const float* input,
    const float* bias,
    float* output,
    int rows,
    int cols
) {
    int input_size = rows * cols * sizeof(float);
    int bias_size = cols * sizeof(float);

    float* d_input = nullptr;
    float* d_bias = nullptr;
    float* d_output = nullptr;

    // Allocate device memory.
    cudaMalloc((void**)&d_input, input_size);
    cudaMalloc((void**)&d_bias, bias_size);
    cudaMalloc((void**)&d_output, input_size);

    // Copy input data from host to device.
    cudaMemcpy(
        d_input,
        input,
        input_size,
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        d_bias,
        bias,
        bias_size,
        cudaMemcpyHostToDevice
    );

    // Launch one fused kernel for bias addition + ReLU.
    int total = rows * cols;
    int threads_per_block = 256;
    int blocks = (total + threads_per_block - 1) / threads_per_block;

    kernel<<<blocks, threads_per_block>>>(
        d_input,
        d_bias,
        d_output,
        rows,
        cols
    );

    // Copy the result back to the host.
    cudaMemcpy(
        output,
        d_output,
        input_size,
        cudaMemcpyDeviceToHost
    );

    // Free device memory.
    cudaFree(d_input);
    cudaFree(d_bias);
    cudaFree(d_output);
}