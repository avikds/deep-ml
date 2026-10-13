#include <cuda_runtime.h>
#include <math.h>

__global__ void attention_kernel(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int S,
    int D
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    __shared__ float scores[16];
    __shared__ float probs[16];
    __shared__ float row_max;
    __shared__ float row_sum;

    // Each thread computes one query-key score.
    if (tid < S) {
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) {
            dot += Q[row * D + d] * K[tid * D + d];
        }
        scores[tid] = dot / sqrtf((float)D);
    }
    __syncthreads();

    // Stable row-wise softmax.
    if (tid == 0) {
        row_max = scores[0];
        for (int j = 1; j < S; ++j) {
            row_max = fmaxf(row_max, scores[j]);
        }

        row_sum = 0.0f;
        for (int j = 0; j < S; ++j) {
            probs[j] = expf(scores[j] - row_max);
            row_sum += probs[j];
        }

        for (int j = 0; j < S; ++j) {
            probs[j] /= row_sum;
        }
    }
    __syncthreads();

    // Each thread computes one output feature.
    if (tid < D) {
        float value = 0.0f;
        for (int j = 0; j < S; ++j) {
            value += probs[j] * V[j * D + tid];
        }
        O[row * D + tid] = value;
    }
}

void solve(const float* Q, const float* K, const float* V, float* O, int S, int D) {
    const size_t bytes = static_cast<size_t>(S) * D * sizeof(float);

    float* d_Q = nullptr;
    float* d_K = nullptr;
    float* d_V = nullptr;
    float* d_O = nullptr;

    cudaMalloc(&d_Q, bytes);
    cudaMalloc(&d_K, bytes);
    cudaMalloc(&d_V, bytes);
    cudaMalloc(&d_O, bytes);

    cudaMemcpy(d_Q, Q, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, bytes, cudaMemcpyHostToDevice);

    // One block per query row. S and D are at most 16.
    attention_kernel<<<S, 16>>>(d_Q, d_K, d_V, d_O, S, D);

    cudaMemcpy(O, d_O, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
}