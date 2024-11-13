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
    __shared__ accT max_val;
    __shared__ accT row_sum;
    __shared__ accT prev_val;

    if (threadIdx.x == 0) {
        max_val = -340282346638528859811704183484516925440.0f;
        row_sum = 0;
    }

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

    for (int i = 0; i < seq_len; i ++) {
        accT value = q_buf[threadIdx.x] * k_buf[i * embed_dim + threadIdx.x] * scale;
        __syncwarp();
        accT sum = tree_reduction_sum(value);
        __syncthreads();

        if (lane_id == 0) {
            atomicAdd(&qk_buf[i], sum);
        }

        __syncthreads();
    }

    if (threadIdx.x == 0) {
        for (int i = 0; i < seq_len; i++) {
            max_val = max(max_val, static_cast<accT>(qk_buf[i]));
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        for (int i = 0; i < seq_len; i ++) {
            row_sum += __expf(static_cast<accT>(qk_buf[i]) - max_val);
        }
    }

    __syncthreads();

    for (int i = 0; i < seq_len; i ++) {
        accT attention_weight = __expf(static_cast<accT>(qk_buf[i]) - max_val)/row_sum;
        output_buf[threadIdx.x] += static_cast<T>(attention_weight) * v_buf[i * embed_dim + threadIdx.x];
    }


    output[head_id * seq_len * embed_dim + seq_id * embed_dim + threadIdx.x] = output_buf[threadIdx.x];

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
