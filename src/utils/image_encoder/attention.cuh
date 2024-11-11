#ifndef ATTENTION_CUH
#define ATTENTION_CUH

#include "utils/common.cuh"
#include "utils/gpu_utils.cuh"


template <typename T, int embed_dim>
__global__ void flash_attention_kernel(const T* query, const T* key, const T* value, T* output, int seq_len) 
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    __shared__ T k_buf[seq_len * embed_dim];
    __shared__ T v_buf[seq_len * embed_dim];
    __shared__ T q_buf[embed_dim];

    __shared__ T qk_buf[seq_len];

    q_buf[threadIdx.x] = query[threadIdx.x];

    for (int i = 0; i < seq_len; i ++) {
         k_buf[i * embed_dim + threadIdx.x] = key[i * embed_dim + threadIdx.x];
         v_buf[i * embed_dim + threadIdx.x] = value[i * embed_dim + threadIdx.x];
    }

    T sum = 0;
    T max_qk = -INFINITY;

    for (int i = 0; i < seq_len; i ++) {
        qk_buf[i] = q_buf[threadIdx.x] * k_buf[i * embed_dim + threadIdx.x];
        max_qk = max(max_qk, qk_buf[i]);
    }

    for (int i = 0; i < seq_len; i ++) {
        qk_buf[i] = exp(qk_buf[i] - max_qk);
    }





}


#endif
