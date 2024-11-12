#ifndef ATTENTION_CUH
#define ATTENTION_CUH

#include "utils/common.h"
#include "utils/gpu_utils.cuh"

#define NEG_INFINITY __int_as_float(0xff800000)


template <typename T, typename accT, int embed_dim, int seq_len>
__global__ void flash_attention_kernel(const T* query, const T* key, const T* value, T* output, T scale) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int seq_id = blockIdx.x;
    int head_id = blockIdx.y;
    int lane_id = threadIdx.x % WARP_SIZE;

    __shared__ T k_buf[seq_len * embed_dim];
    __shared__ T v_buf[seq_len * embed_dim];
    __shared__ T q_buf[embed_dim];
    __shared__ T qk_buf[seq_len];
    __shared__ T output_buf[embed_dim];

    q_buf[threadIdx.x] = query[head_id * seq_len * embed_dim + seq_id * embed_dim + threadIdx.x];
    output_buf[threadIdx.x] = 0;

    for (int i = 0; i < seq_len; i ++) {
         k_buf[i * embed_dim + threadIdx.x] = key[head_id * seq_len * embed_dim + i * embed_dim + threadIdx.x];
         v_buf[i * embed_dim + threadIdx.x] = value[head_id * seq_len * embed_dim + i * embed_dim + threadIdx.x];

         if (i == 0) {
            qk_buf[i] = 0;
         }
    }

    __syncthreads();

    accT row_sum = 0;
    accT max_val = 340282346638528859811704183484516925440.0f;
    max_val = -max_val;
    accT prev_val;

    for (int i = 0; i < seq_len; i ++) {
        accT value = q_buf[threadIdx.x] * k_buf[i * embed_dim + threadIdx.x];
        __syncwarp();
        accT sum = tree_reduction_sum(value);
        __syncthreads();

        if (lane_id == 0) {
            atomicAdd(&qk_buf[i], sum);
        }

        __syncthreads();

        qk_buf[i] = qk_buf[i] * scale;

        max_val = max(max_val, qk_buf[i]);

        accT p = __expf(qk_buf[i] - max_val);

        if (i == 0) {
            row_sum = p;
        } else {
            row_sum = __expf(prev_val - max_val) * row_sum  + p;
        }

        output_buf[threadIdx.x] += (__expf(prev_val - max_val) * output_buf[threadIdx.x]) + (p * v_buf[i * embed_dim + threadIdx.x]);
        prev_val = max_val;
    }

    output[head_id * seq_len * embed_dim + seq_id * embed_dim + threadIdx.x] = output_buf[threadIdx.x]/row_sum;

}


template <typename T, typename accT, int embed_dim, int seq_len>
void flash_attention_kernel_wrapper(const T* query, const T* key, const T* value, T* output, int num_heads) {

    T* d_query, *d_key, *d_value, *d_output;
    int total_size = embed_dim * seq_len * num_heads;


    gpuErrchk(cudaMalloc(&d_query, sizeof(T) * total_size));
    gpuErrchk(cudaMalloc(&d_key, sizeof(T) * total_size));
    gpuErrchk(cudaMalloc(&d_value, sizeof(T) * total_size));
    gpuErrchk(cudaMalloc(&d_output, sizeof(T) * total_size));

    gpuErrchk(cudaMemcpy(d_query, query, sizeof(T) * total_size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_key, key, sizeof(T) * total_size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_value, value, sizeof(T) * total_size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemset(d_output, 0, sizeof(T) * total_size));

    T scale = 1.0f/sqrtf(embed_dim);

    dim3 block_size(embed_dim);
    dim3 grid_size(seq_len, num_heads);

    flash_attention_kernel<T, accT, embed_dim, seq_len><<<grid_size, block_size>>>(d_query, d_key, d_value, d_output, scale);

    gpuErrchk(cudaMemcpy(output, d_output, sizeof(T) * total_size, cudaMemcpyDeviceToHost));

    gpuErrchk(cudaFree(d_query));
    gpuErrchk(cudaFree(d_key));
    gpuErrchk(cudaFree(d_value));
    gpuErrchk(cudaFree(d_output));
}

#endif
