#include <cuda_runtime.h>
#include <math.h>

#define BC 4

__global__ void tiled_flash_attention_kernel(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int S,
    int D
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    __shared__ float scores[BC];
    __shared__ float probs[BC];
    __shared__ float out_acc[16];

    __shared__ float m_state;
    __shared__ float l_state;
    __shared__ float alpha;
    __shared__ float m_new;

    // Initialize the online softmax state for this query row.
    if (tid == 0) {
        m_state = -INFINITY;
        l_state = 0.0f;
    }
    if (tid < D) {
        out_acc[tid] = 0.0f;
    }
    __syncthreads();

    const float scale = rsqrtf((float)D);

    // Stream the KV cache in tiles of four tokens.
    for (int start = 0; start < S; start += BC) {
        // Compute one score per tile position.
        if (tid < BC) {
            int j = start + tid;

            if (j < S) {
                float dot = 0.0f;
                for (int d = 0; d < D; ++d) {
                    dot += Q[row * D + d] * K[j * D + d];
                }
                scores[tid] = dot * scale;
            } else {
                scores[tid] = -INFINITY;
            }
        }
        __syncthreads();

        // Update the running maximum, normalization sum, and tile weights.
        if (tid == 0) {
            float tile_max = scores[0];
            for (int j = 1; j < BC; ++j) {
                tile_max = fmaxf(tile_max, scores[j]);
            }

            m_new = fmaxf(m_state, tile_max);
            alpha = expf(m_state - m_new);

            float tile_sum = 0.0f;
            for (int j = 0; j < BC; ++j) {
                probs[j] = expf(scores[j] - m_new);
                tile_sum += probs[j];
            }

            l_state = l_state * alpha + tile_sum;
            m_state = m_new;
        }
        __syncthreads();

        // Update the unnormalized output accumulator:
        // o' = o * exp(m - m') + sum_j exp(s_j - m') * V_j.
        if (tid < D) {
            float tile_out = 0.0f;

            for (int j = 0; j < BC; ++j) {
                int key_idx = start + j;

                if (key_idx < S) {
                    tile_out += probs[j] * V[key_idx * D + tid];
                }
            }

            out_acc[tid] = out_acc[tid] * alpha + tile_out;
        }
        __syncthreads();
    }

    // Normalize the final output.
    if (tid < D) {
        O[row * D + tid] = out_acc[tid] / l_state;
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

    // One block per query row; 32 threads support D <= 16 and BC = 4.
    tiled_flash_attention_kernel<<<S, 32>>>(
        d_Q, d_K, d_V, d_O, S, D
    );

    cudaMemcpy(O, d_O, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
}