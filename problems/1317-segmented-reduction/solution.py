#include <cuda_runtime.h>

__global__ void segmented_reduce_kernel(
    const float* values,
    const int* flags,
    float* output,
    int n
) {
    __shared__ float sum_a[1024];
    __shared__ float sum_b[1024];
    __shared__ int head_a[1024];
    __shared__ int head_b[1024];

    __shared__ int scan_a[1024];
    __shared__ int scan_b[1024];

    const int tid = threadIdx.x;

    // Initial segmented-scan state.
    sum_a[tid] = values[tid];
    head_a[tid] = flags[tid];
    __syncthreads();

    // Parallel segmented inclusive scan.
    float* sum_src = sum_a;
    float* sum_dst = sum_b;
    int* head_src = head_a;
    int* head_dst = head_b;

    for (int offset = 1; offset < n; offset <<= 1) {
        float cur_sum = sum_src[tid];
        int cur_head = head_src[tid];

        float new_sum = cur_sum;
        int new_head = cur_head;

        if (tid >= offset) {
            float prev_sum = sum_src[tid - offset];
            int prev_head = head_src[tid - offset];

            // If the current aggregate already contains a segment start,
            // it supersedes everything to its left.
            if (cur_head) {
                new_sum = cur_sum;
            } else {
                new_sum = cur_sum + prev_sum;
            }

            new_head = cur_head | prev_head;
        }

        sum_dst[tid] = new_sum;
        head_dst[tid] = new_head;
        __syncthreads();

        float* tmp_sum = sum_src;
        sum_src = sum_dst;
        sum_dst = tmp_sum;

        int* tmp_head = head_src;
        head_src = head_dst;
        head_dst = tmp_head;

        __syncthreads();
    }

    // Inclusive scan of segment-start flags gives each element's
    // segment number (1-based).
    scan_a[tid] = flags[tid];
    __syncthreads();

    int* scan_src = scan_a;
    int* scan_dst = scan_b;

    for (int offset = 1; offset < n; offset <<= 1) {
        int value = scan_src[tid];

        if (tid >= offset) {
            value += scan_src[tid - offset];
        }

        scan_dst[tid] = value;
        __syncthreads();

        int* tmp = scan_src;
        scan_src = scan_dst;
        scan_dst = tmp;

        __syncthreads();
    }

    // At the end of each segment, the segmented prefix sum is
    // the complete segment sum. Pack it using the segment number.
    if (tid == n - 1 || flags[tid + 1] == 1) {
        int segment_id = scan_src[tid] - 1;
        output[segment_id] = sum_src[tid];
    }
}

void solve(const float* values, const int* flags, float* output, int n) {
    float* d_values = nullptr;
    int* d_flags = nullptr;
    float* d_output = nullptr;

    const size_t value_bytes = static_cast<size_t>(n) * sizeof(float);
    const size_t flag_bytes = static_cast<size_t>(n) * sizeof(int);

    cudaMalloc(&d_values, value_bytes);
    cudaMalloc(&d_flags, flag_bytes);
    cudaMalloc(&d_output, value_bytes);

    cudaMemcpy(
        d_values, values, value_bytes,
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        d_flags, flags, flag_bytes,
        cudaMemcpyHostToDevice
    );

    segmented_reduce_kernel<<<1, n>>>(
        d_values,
        d_flags,
        d_output,
        n
    );

    cudaMemcpy(
        output, d_output, value_bytes,
        cudaMemcpyDeviceToHost
    );

    cudaFree(d_values);
    cudaFree(d_flags);
    cudaFree(d_output);
}