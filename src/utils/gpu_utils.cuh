#ifndef GPU_UTILS_CUH
#define GPU_UTILS_CUH

#include <cstdio>

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
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