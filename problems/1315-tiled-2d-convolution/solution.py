#include <cuda_runtime.h>

__global__ void tiled_conv2d_kernel(
    const float* input,
    const float* kernel,
    float* output,
    int H,
    int W,
    int radius
) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int out_x = blockIdx.x * blockDim.x + tx;
    const int out_y = blockIdx.y * blockDim.y + ty;

    const int tile_w = blockDim.x + 2 * radius;
    const int tile_h = blockDim.y + 2 * radius;

    extern __shared__ float tile[];

    // Load the output tile plus its halo cooperatively.
    const int tid = ty * blockDim.x + tx;
    const int nthreads = blockDim.x * blockDim.y;
    const int tile_size = tile_w * tile_h;

    for (int idx = tid; idx < tile_size; idx += nthreads) {
        const int sy = idx / tile_w;
        const int sx = idx % tile_w;

        const int gy = blockIdx.y * blockDim.y + sy - radius;
        const int gx = blockIdx.x * blockDim.x + sx - radius;

        if (gy >= 0 && gy < H && gx >= 0 && gx < W) {
            tile[idx] = input[gy * W + gx];
        } else {
            tile[idx] = 0.0f;
        }
    }

    __syncthreads();

    if (out_y >= H || out_x >= W) {
        return;
    }

    float acc = 0.0f;

    for (int u = -radius; u <= radius; ++u) {
        for (int v = -radius; v <= radius; ++v) {
            const int sy = ty + u + radius;
            const int sx = tx + v + radius;

            const float xval = tile[sy * tile_w + sx];

            const int ky = u + radius;
            const int kx = v + radius;
            const float kval = kernel[ky * (2 * radius + 1) + kx];

            acc += xval * kval;
        }
    }

    output[out_y * W + out_x] = acc;
}

void solve(const float* input, const float* kernel, float* output, int H, int W, int radius) {
    float* d_input = nullptr;
    float* d_kernel = nullptr;
    float* d_output = nullptr;

    const size_t input_bytes = static_cast<size_t>(H) * W * sizeof(float);
    const size_t kernel_size =
        static_cast<size_t>(2 * radius + 1) * (2 * radius + 1);
    const size_t kernel_bytes = kernel_size * sizeof(float);

    cudaMalloc(&d_input, input_bytes);
    cudaMalloc(&d_kernel, kernel_bytes);
    cudaMalloc(&d_output, input_bytes);

    cudaMemcpy(d_input, input, input_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernel_bytes, cudaMemcpyHostToDevice);

    const dim3 block(16, 16);
    const dim3 grid(
        (W + block.x - 1) / block.x,
        (H + block.y - 1) / block.y
    );

    const size_t shared_bytes =
        static_cast<size_t>(block.x + 2 * radius) *
        static_cast<size_t>(block.y + 2 * radius) *
        sizeof(float);

    tiled_conv2d_kernel<<<grid, block, shared_bytes>>>(
        d_input,
        d_kernel,
        d_output,
        H,
        W,
        radius
    );

    cudaMemcpy(output, d_output, input_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
}