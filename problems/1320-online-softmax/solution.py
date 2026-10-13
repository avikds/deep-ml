#include <cuda_runtime.h>
#include <math.h>

__global__ void online_softmax_kernel(
    const float* input,
    float* output,
    int rows,
    int cols
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    __shared__ float sdata[32];
    __shared__ float global_max;
    __shared__ float global_sum;

    if (tid == 0) {
        global_max = -INFINITY;
        global_sum = 0.0f;
    }
    __syncthreads();

    // Process the row in tiles of 32 elements.
    for (int start = 0; start < cols; start += 32) {
        const int col = start + tid;
        const bool valid = col < cols;

        float xval = valid ? input[row * cols + col] : -INFINITY;

        // Local maximum for this tile.
        sdata[tid] = xval;
        __syncthreads();

        for (int offset = 16; offset > 0; offset >>= 1) {
            if (tid < offset) {
                sdata[tid] = fmaxf(sdata[tid], sdata[tid + offset]);
            }
            __syncthreads();
        }

        const float tile_max = sdata[0];

        // Local sum of exponentials relative to the tile maximum.
        sdata[tid] = valid ? expf(xval - tile_max) : 0.0f;
        __syncthreads();

        for (int offset = 16; offset > 0; offset >>= 1) {
            if (tid < offset) {
                sdata[tid] += sdata[tid + offset];
            }
            __syncthreads();
        }

        const float tile_sum = sdata[0];

        // Online recurrence merges this tile with previous tiles.
        if (tid == 0) {
            const float new_max = fmaxf(global_max, tile_max);

            global_sum =
                global_sum * expf(global_max - new_max) +
                tile_sum * expf(tile_max - new_max);

            global_max = new_max;
        }
        __syncthreads();
    }

    // Second pass: normalize each element using the final statistics.
    for (int col = tid; col < cols; col += 32) {
        output[row * cols + col] =
            expf(input[row * cols + col] - global_max) / global_sum;
    }
}

void solve(const float* input, float* output, int rows, int cols) {
    const size_t bytes =
        static_cast<size_t>(rows) * cols * sizeof(float);

    float* d_input = nullptr;
    float* d_output = nullptr;

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    cudaMemcpy(d_input, input, bytes, cudaMemcpyHostToDevice);

    // One block per row, with 32 threads per block.
    online_softmax_kernel<<<rows, 32>>>(
        d_input, d_output, rows, cols
    );

    cudaMemcpy(output, d_output, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}