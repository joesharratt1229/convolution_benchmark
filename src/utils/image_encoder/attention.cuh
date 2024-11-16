#ifndef ATTENTION_CUH
#define ATTENTION_CUH

#include "utils/common.h"
#include "utils/gpu_utils.cuh"

#define NEG_INFINITY __int_as_float(0xff800000)

template <typename accT>
__device__ inline accT calculate_block_max(accT warp_max, int tid, int seq_len) {
    __shared__ accT warpMaxes[NumWarps];

    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    if (lane_id == 0) {
        warpMaxes[warp_id] = warp_max;
    }

    __syncthreads();

    accT block_max = get_lowest<accT>();
    if (warp_id == 0) {
        int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
        accT thread_val = (lane_id < num_warps) ? warpMaxes[lane_id] : get_lowest<accT>();
        block_max = tree_reduction_max(thread_val);

    }

    if (warp_id == 0 && lane_id == 0) {
        warpMaxes[0] = block_max;
    }

    __syncthreads();
    
    return warpMaxes[0];
}

template <typename accT>
__device__ inline accT calculate_block_sum(accT warp_sum, int tid) {
    __shared__ accT warpSums[NumWarps];
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    if (lane_id == 0) {
        warpSums[warp_id] = warp_sum;
    }

    __syncthreads();

    accT block_sum = 0;
    if (warp_id == 0) {
        int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
        accT thread_val = (lane_id < num_warps) ? warpSums[lane_id] : 0;
        block_sum = tree_reduction_sum(thread_val);
    }

    if (warp_id == 0 && lane_id == 0) {
        warpSums[0] = block_sum;
    }

    __syncthreads();
    
    return warpSums[0];
}


template <typename T, typename accT, int embed_dim, int seq_len, int warps_per_block>
__global__ void scalable_flash_attention_kernel(const T* query, const T* key, const T* value, T* output, accT scale, int Tc) {

    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int query_seq_idx = (blockIdx.x * Tc * warps_per_block) + (warp_id * Tc);
    int head_id = blockIdx.y;
    int thread_seq_idx = (lane_id*Tc);

    __shared__ T k_buf[seq_len * embed_dim];
    __shared__ T v_buf[seq_len * embed_dim];
    __shared__ accT qk_buf[warps_per_block * seq_len];

    if (threadIdx.x < seq_len) {
        for (int i = 0; i < embed_dim; i ++) {
            k_buf[threadIdx.x * embed_dim + i] = key[head_id * seq_len * embed_dim + threadIdx.x * embed_dim + i];
            v_buf[threadIdx.x * embed_dim + i] = value[head_id * seq_len * embed_dim + threadIdx.x * embed_dim + i];
        }
    }

    __syncthreads();

    for (int q = 0; q < Tc; q ++)
    {

        for (int seq_idx = 0; seq_idx < Tc; seq_idx ++) {
            qk_buf[warp_id * seq_len + thread_seq_idx + seq_idx] = 0;

            for (int d = 0; d < embed_dim; d ++) {
                qk_buf[warp_id * seq_len + thread_seq_idx + seq_idx] += static_cast<accT>(query[head_id * seq_len * embed_dim + (query_seq_idx + q) * embed_dim + d]) * static_cast<accT>(k_buf[(thread_seq_idx + seq_idx) * embed_dim + d]) * scale;
            }
        }

        accT thread_max = get_lowest<accT>();
        accT thread_sum = 0;

        for (int seq_idx = 0; seq_idx < Tc; seq_idx ++) {
            thread_max = fmaxf(thread_max, qk_buf[warp_id * seq_len + thread_seq_idx + seq_idx]);
        }

        for (int seq_idx = 0; seq_idx < Tc; seq_idx ++) {
            qk_buf[warp_id * seq_len + thread_seq_idx + seq_idx] = __expf(qk_buf[warp_id * seq_len + thread_seq_idx + seq_idx] - thread_max);
            thread_sum += qk_buf[warp_id * seq_len + thread_seq_idx + seq_idx];
        }

        accT warp_sum = tree_reduction_sum(thread_sum);
        accT warp_max = tree_reduction_max(thread_max);

        accT rescaled_sum = warp_sum * __expf(warp_max - thread_max);

        for (int seq_idx = 0; seq_idx < Tc; seq_idx ++) {
            qk_buf[warp_id * seq_len + thread_seq_idx + seq_idx] *= __expf(warp_max - thread_max);
            qk_buf[warp_id * seq_len + thread_seq_idx + seq_idx] /= rescaled_sum;
        }

        for (int d = 0; d < embed_dim; d ++) {
            accT output_val = 0;
            for (int seq_idx = 0; seq_idx < Tc; seq_idx ++) {
                output_val += qk_buf[warp_id * seq_len + thread_seq_idx + seq_idx] * v_buf[(thread_seq_idx + seq_idx) * embed_dim + d];
            }

            warp_sum = tree_reduction_sum(output_val);

            if (threadIdx.x == 0) {
                output[head_id * seq_len * embed_dim + (query_seq_idx + q) * embed_dim + d] = static_cast<T>(warp_sum);
            }
        }   
    }
}


template <typename T, typename accT, int embed_dim, int seq_len, int num_heads>
void scalable_flash_attention_kernel_wrapper(const T* query, const T* key, const T* value, T* output, accT scale) {
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

    const int warps_per_block = (seq_len) / WARP_SIZE;
    const int Tc = 2;

    dim3 block_size(warps_per_block * WARP_SIZE);
    dim3 grid_size(seq_len/(Tc * warps_per_block), num_heads);

    scalable_flash_attention_kernel<T, accT, embed_dim, seq_len, warps_per_block><<<grid_size, block_size>>>(d_query, d_key, d_value, d_output, scale, Tc);

    gpuErrchk(cudaMemcpy(output, d_output, sizeof(T) * total_size, cudaMemcpyDeviceToHost));

    gpuErrchk(cudaFree(d_query));
    gpuErrchk(cudaFree(d_key));
    gpuErrchk(cudaFree(d_value));
    gpuErrchk(cudaFree(d_output));
}



template <typename T, typename accT, int embed_dim, int seq_len>
__global__ void flash_attention_kernel(const T* query, const T* key, const T* value, T* output, accT scale) {

    int query_seq_idx = (blockIdx.x * Tc);
    int head_id = blockIdx.y;
    int seq_id = threadIdx.x;

    __shared__ T k_buf[seq_len * embed_dim];
    __shared__ T v_buf[seq_len * embed_dim];
    __shared__ accT qk_buf[seq_len];

    for (int i = 0; i < embed_dim; i ++) {
        k_buf[seq_id * embed_dim + i] = key[head_id * seq_len * embed_dim + seq_id * embed_dim + i];
        v_buf[seq_id * embed_dim + i] = value[head_id * seq_len * embed_dim + seq_id * embed_dim + i];
    }

    __syncthreads();

    for (int q = 0; q < Tc; q++) {
        qk_buf[seq_id] = 0;

        for (int i = 0; i < embed_dim; i ++) {
            qk_buf[seq_id] += static_cast<accT>(query[head_id * seq_len * embed_dim + (query_seq_idx + q) * embed_dim + i]) * static_cast<accT>(k_buf[seq_id * embed_dim + i]) * scale;
        }

        accT warp_max = tree_reduction_max(qk_buf[seq_id]);
        accT p = __expf(qk_buf[seq_id] - warp_max);
        accT block_max = calculate_block_max(warp_max, threadIdx.x, seq_len);
        accT softmax_val = p * __expf(warp_max - block_max);
        accT warp_recaled_sum = tree_reduction_sum(softmax_val);
        accT rescaled_sum = warp_recaled_sum * __expf(warp_max - block_max);
        accT rescaled_sum_block = calculate_block_sum(rescaled_sum, threadIdx.x);
        qk_buf[seq_id] = softmax_val/(rescaled_sum_block);
        accT output_val;

        for (int d = 0; d < embed_dim; d ++) {
            output_val = qk_buf[seq_id] * v_buf[seq_id * embed_dim + d];
            accT warp_sum = tree_reduction_sum(output_val);
            accT block_sum = calculate_block_sum(warp_sum, threadIdx.x);

            if (threadIdx.x == 0) {
                output[head_id * seq_len * embed_dim + (query_seq_idx + q) * embed_dim + d] = static_cast<T>(block_sum);
            }
        }
    } 
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

    const accT scale = 1.0f/sqrtf(embed_dim);
    const int Bc = seq_len/Tc;

    dim3 block_size(seq_len);
    dim3 grid_size(Bc, num_heads);

    flash_attention_kernel<T, accT, embed_dim, seq_len><<<grid_size, block_size>>>(d_query, d_key, d_value, d_output, scale);

    gpuErrchk(cudaMemcpy(output, d_output, sizeof(T) * total_size, cudaMemcpyDeviceToHost));

    gpuErrchk(cudaFree(d_query));
    gpuErrchk(cudaFree(d_key));
    gpuErrchk(cudaFree(d_value));
    gpuErrchk(cudaFree(d_output));
}

#endif
