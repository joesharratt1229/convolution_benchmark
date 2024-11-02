#ifndef GPU_UTILS_CUH
#define GPU_UTILS_CUH

#include <cstdio>

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