#ifndef ATTENTION_CUH
#define ATTENTION_CUH

#include "utils/common.h"
#include "utils/gpu_utils.cuh"

#define NEG_INFINITY __int_as_float(0xff800000)

template <typename accT, int block_y_dim>
__device__ inline accT calculate_block_max_row(accT val, int tid_x, int tid_y) {
    const int lane_id = tid_x % WARP_SIZE;
    const int warp_id = tid_x / WARP_SIZE;
    const int warps_per_row = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    
    __shared__ accT warpMaxes[block_y_dim * NumWarps];  
    accT warp_max = tree_reduction_max(val);
    
    const int warp_slot = tid_y * warps_per_row + warp_id;
    if (lane_id == 0) {
        warpMaxes[warp_slot] = warp_max;
    }
    __syncthreads();
    
    accT block_max;
    if (warp_id == 0) {  
        const int row_start = tid_y * warps_per_row;
        block_max = (lane_id < warps_per_row) ? warpMaxes[row_start + lane_id] : 0;
        block_max = tree_reduction_max(block_max);
        
        if (lane_id == 0) {
            warpMaxes[tid_y * warps_per_row] = block_max;  
        }
    }
    __syncthreads();
    
    return warpMaxes[tid_y * warps_per_row];
}

template <typename accT, int block_y_dim>
__device__ inline accT calculate_block_sum_row(accT val, int tid_x, int tid_y) {
    const int lane_id = tid_x % WARP_SIZE;
    const int warp_id = tid_x / WARP_SIZE;
    const int warps_per_row = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    __shared__ accT warpSums[block_y_dim * NumWarps];  
    
    accT warp_sum = tree_reduction_sum(val);
    
    const int warp_slot = tid_y * warps_per_row + warp_id;
    if (lane_id == 0) {
        warpSums[warp_slot] = warp_sum;
    }
    __syncthreads();
    
    accT block_sum;
    if (warp_id == 0) {  
        const int row_start = tid_y * warps_per_row;
        block_sum = (lane_id < warps_per_row) ? warpSums[row_start + lane_id] : 0;
        block_sum = tree_reduction_sum(block_sum);
        
        if (lane_id == 0) {
            warpSums[tid_y * warps_per_row] = block_sum;  
        }
    }
    __syncthreads();
    
    return warpSums[tid_y * warps_per_row];
}


template <typename T, typename accT, int embed_dim, int seq_len, int block_y_dim>
__global__ void flash_attention_kernel(const T* query, const T* key, const T* value, T* output, accT scale) {

    int block_query_index = threadIdx.x/seq_len;
    int seq_id = threadIdx.x % seq_len;
    int query_seq_idx = (blockIdx.x * Tc * blockDim.y) + (block_query_index * Tc);
    int head_id = blockIdx.y;
    //size_t block_mem_size = NumWarps * blockDim.y * sizeof(accT);

    __shared__ T k_buf[seq_len * embed_dim];
    __shared__ T v_buf[seq_len * embed_dim];
    __shared__ accT qk_buf[seq_len];

    for (int i = 0; i < embed_dim; i ++) {
        if (threadIdx.x < seq_len){
            k_buf[seq_id * embed_dim + i] = key[head_id * seq_len * embed_dim + seq_id * embed_dim + i];
            v_buf[seq_id * embed_dim + i] = value[head_id * seq_len * embed_dim + seq_id * embed_dim + i];
        }
    }

    __syncthreads();

    for (int q = 0; q < Tc; q++) {
        qk_buf[seq_id] = 0;

        for (int i = 0; i < embed_dim; i ++) {
            qk_buf[seq_id] += static_cast<accT>(query[head_id * seq_len * embed_dim + (query_seq_idx + q) * embed_dim + i]) * static_cast<accT>(k_buf[seq_id * embed_dim + i]) * scale;
        }

        accT warp_max = tree_reduction_max(qk_buf[seq_id]);
        accT p = __expf(qk_buf[seq_id] - warp_max);
        accT block_max = calculate_block_max_row<accT, block_y_dim>(warp_max, threadIdx.x, threadIdx.y);
        accT softmax_val = p * __expf(warp_max - block_max);
        accT warp_recaled_sum = tree_reduction_sum(softmax_val);
        accT rescaled_sum = warp_recaled_sum * __expf(warp_max - block_max);
        accT rescaled_sum_block = calculate_block_sum_row<accT, block_y_dim>(rescaled_sum, threadIdx.x, threadIdx.y);
        qk_buf[seq_id] = softmax_val/(rescaled_sum_block);
        accT output_val;

        for (int d = 0; d < embed_dim; d ++) {
            output_val = qk_buf[seq_id] * v_buf[seq_id * embed_dim + d];
            accT warp_sum = tree_reduction_sum(output_val);
            accT block_sum = calculate_block_sum_row<accT, block_y_dim>(warp_sum, threadIdx.x, threadIdx.y);

            if (threadIdx.x == 0) {
                output[head_id * seq_len * embed_dim + (query_seq_idx + q) * embed_dim + d] = static_cast<T>(block_sum);
            }
        }
    } 
}



template <typename T, typename accT, int embed_dim, int seq_len>
void flash_attention_kernel_wrapper(const T* query, const T* key, const T* value, T* output, int num_heads) {

    cudaDeviceProp prop;
    gpuErrchk(cudaGetDeviceProperties(&prop, 0));
    constexpr int block_y_dim = max_threads_per_block/seq_len;

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

    const accT scale = 1.0f/sqrtf(embed_dim);
    const int Bc = seq_len/(Tc * block_y_dim);

    dim3 block_size(seq_len, block_y_dim);
    dim3 grid_size(Bc, num_heads);

    flash_attention_kernel<T, accT, embed_dim, seq_len, block_y_dim><<<grid_size, block_size>>>(d_query, d_key, d_value, d_output, scale);

    gpuErrchk(cudaMemcpy(output, d_output, sizeof(T) * total_size, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(d_query));
    gpuErrchk(cudaFree(d_key));
    gpuErrchk(cudaFree(d_value));
    gpuErrchk(cudaFree(d_output));
}

#endif
