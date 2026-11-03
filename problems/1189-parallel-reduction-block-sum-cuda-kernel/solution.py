#include <cuda_runtime.h>

#define THREADS 256

__global__ void kernel(const float* in, float* partial, int N) {
    __shared__ float sdata[THREADS];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * (blockDim.x * 2) + tid;

    float sum = 0.0f;

    while (idx < N) {
        sum += in[idx];

        if (idx + THREADS < N)
            sum += in[idx + THREADS];

        idx += gridDim.x * blockDim.x * 2;
    }

    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2;
         stride > 0; stride >>= 1) {
        if (tid < stride)
            sdata[tid] += sdata[tid + stride];

        __syncthreads();
    }

    if (tid == 0)
        partial[blockIdx.x] = sdata[0];
}

__global__ void final_reduce(const float* partial, float* output,
                             int num_blocks) {
    __shared__ float sdata[THREADS];

    unsigned int tid = threadIdx.x;
    float sum = 0.0f;

    for (int i = tid; i < num_blocks; i += blockDim.x)
        sum += partial[i];

    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2;
         stride > 0; stride >>= 1) {
        if (tid < stride)
            sdata[tid] += sdata[tid + stride];

        __syncthreads();
    }

    if (tid == 0)
        output[0] = sdata[0];
}

void solve(const float* input, float* output, int N) {
    if (N <= 0) {
        *output = 0.0f;
        return;
    }

    int num_blocks = (N + THREADS * 2 - 1) / (THREADS * 2);

    float *d_in, *d_partial, *d_out;

    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_partial, num_blocks * sizeof(float));
    cudaMalloc(&d_out, sizeof(float));

    cudaMemcpy(d_in, input, N * sizeof(float),
               cudaMemcpyHostToDevice);

    kernel<<<num_blocks, THREADS>>>(d_in, d_partial, N);

    final_reduce<<<1, THREADS>>>(d_partial, d_out, num_blocks);

    cudaMemcpy(output, d_out, sizeof(float),
               cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_partial);
    cudaFree(d_out);
}