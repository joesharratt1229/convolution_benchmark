#ifndef GPU_UTILS_CUH
#define GPU_UTILS_CUH

#define FULL_MASK 0xffffffff

#include <cstdio>
#include <cuda_fp16.h>
#include <limits>

constexpr static int WARP_SIZE = 32;
constexpr static int THREADS_PER_BLOCK = 1024;
constexpr static int NUM_WARPS = THREADS_PER_BLOCK / WARP_SIZE;
constexpr static int TC = 2;

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

inline int getSMCount() {
    int deviceId;
    cudaGetDevice(&deviceId);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, deviceId);
    return props.multiProcessorCount;
}

inline dim3 adjust_channel_size(dims dim, dim3 threadDims) {
    int x = (dim.width + threadDims.x - 1) / threadDims.x;
    int y = (dim.height + threadDims.y - 1) / threadDims.y;

    int num_sms = getSMCount();
    int current_blocks = x * y * dim.channel;
    int target_waves = (current_blocks + num_sms - 1) / num_sms;

    return dim3((dim.width + threadDims.x - 1) / threadDims.x, 
               (dim.height + threadDims.y - 1) / threadDims.y, 
               (target_waves*num_sms)/(x*y));
}


template <typename T>
__device__ inline T tree_reduction_sum(T value)
{
    for (int i = 16; i > 0; i /= 2) {
        value += __shfl_down_sync(FULL_MASK, value, i);
    }
    return value;
}

template <typename T>
__device__ inline T tree_reduction_max(T value)
{
    for (int i = 16; i > 0; i /= 2) {
        value = max(value, __shfl_down_sync(FULL_MASK, value, i));
    }
    return value;
}

template <typename accT>
__device__ inline accT get_lowest() {
    if (std::is_same<accT, float>::value) {
        return -__FLT_MAX__;
    } else if (std::is_same<accT, half>::value) {
        return -65504.0f;  // Lowest representable half precision value
    } else if (std::is_same<accT, double>::value) {
        return -__DBL_MAX__;
    }
    return 0;  // Default case
}

namespace cuda_config {
    // For convolution kernel (2D spatial + channel)
    struct StandardConfig {
        static constexpr int BLOCK_X = 32;  // Spatial X dimension
        static constexpr int BLOCK_Y = 32;  // Spatial Y dimension
        static constexpr int BLOCK_Z = 1;   // Channel dimension
        
        static dim3 block_dim() { 
            return dim3(BLOCK_X, BLOCK_Y, BLOCK_Z); 
        }
    };
}

#endif