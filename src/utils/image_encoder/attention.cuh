#ifndef ATTENTION_CUH
#define ATTENTION_CUH

#include "utils/common.h"
#include "utils/gpu_utils.cuh"

#define NEG_INFINITY __int_as_float(0xff800000)

template <typename accT>
__device__ inline accT calculate_block_max(accT val, int tid) {
    __shared__ accT warpMaxes[32];
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    accT warp_max = tree_reduction_max(val, WARP_SIZE);

    if (lane_id == 0) {
        warpMaxes[warp_id] = warp_max;
    }

    __syncthreads();

    accT block_max;

    if (warp_id == 0) {
        accT thread_val = (tid < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) 
            ? warpMaxes[lane_id] 
            : get_lowest<accT>();
        block_max = tree_reduction_max(thread_val, WARP_SIZE);
    }

    if (tid == 0) {
        block_max = tree_reduction_max(warpMaxes, WARP_SIZE);
    }

    return block_max;
}

template <typename T, typename accT, int embed_dim, int seq_len>
__global__ void flash_attention_kernel(const T* query, const T* key, const T* value, T* output) 
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    __shared__ T k_buf[seq_len * embed_dim];
    __shared__ T v_buf[seq_len * embed_dim];
    __shared__ T q_buf[embed_dim];
    __shared__ accT warpMaxes[32];
    __shared__ T qk_buf[seq_len];

    q_buf[threadIdx.x] = query[threadIdx.x];

    for (int i = 0; i < seq_len; i ++) {
         k_buf[i * embed_dim + threadIdx.x] = key[i * embed_dim + threadIdx.x];
         v_buf[i * embed_dim + threadIdx.x] = value[i * embed_dim + threadIdx.x];

         if (lane_id == 0) {
            qk_buf[i] = 0;
         }
    }

    for (int i = 0; i < seq_len; i ++) {
        accT value = q_buf[threadIdx.x] * k_buf[i * embed_dim + threadIdx.x];
        __syncwarp();
        accT sum = tree_reduction_sum(value);
 
       if (lane_id == 0) {
            qk_buf[i] += sum;
        }
    }

    //compute local max within warp
    accT block_max = calculate_block_max(qk_buf[threadIdx.x], tid);

    accT sum = 0;

    for (int i = 0; i < seq_len; i ++) {
        accT p = exp(qk_buf[i] - block_max);
        sum += p;
    }

    accT output_val = 0;

    for (int i = 0; i < seq_len; i ++) {
        accT p = exp(qk_buf[i] - block_max);
        output_val += p * v_buf[i * embed_dim + threadIdx.x] / sum;
    }

    output[threadIdx.x] = output_val;
}

#endif
