#include <cmath>
#include "cuda_runtime.h"

#include "utils/common.h"
#include "utils/gpu_utils.cuh"


#define EPSILON 1e-6
#define TEMPERATURE 10000
#define numPosFeats 128


namespace image_encoder {       

template<typename T, typename accFloatT>
__global__ void pos_embedding_kernel(T* pos_embeds,
                                    accFloatT* d_dimensions_x,
                                    accFloatT* d_dimensions_y,
                                    const int nx,
                                    const int ny)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y ;
    int pos_feat = threadIdx.z + blockIdx.z * blockDim.z;


    accFloatT y_embed = y+1;
    accFloatT x_embed = x+1;

    if (pos_feat < numPosFeats && x == 0 && y == 0)
    {
        accFloatT power_term = 2*(floorf((pos_feat)/2))/numPosFeats;
        d_dimensions_x[pos_feat] = std::pow(TEMPERATURE, power_term);
        d_dimensions_y[pos_feat] = std::pow(TEMPERATURE, power_term);
    }

    __syncthreads();

    if (pos_feat < numPosFeats && x < nx && y < ny)
    {
       const bool is_even = pos_feat%2 == 0;
       const bool is_first_half = pos_feat < numPosFeats/2;
       const accFloatT embed_val = is_first_half ? y_embed : x_embed;
       const accFloatT dim = is_first_half ? d_dimensions_y[pos_feat] : d_dimensions_x[pos_feat];

       accFloatT val = is_even ? std::sin(embed_val / dim) : std::cos(embed_val / dim);
       pos_embeds[pos_feat * (ny * nx) + y * nx + x] = static_cast<T>(val);
    }
}



template<typename T, typename accFloatT>
T* template_pos_embedding(const int nx, const int ny)
{
    T* h_pos_embeds;
    T* d_pos_embeds;
    accFloatT* d_dimensions_x;
    accFloatT* d_dimensions_y;

    cudaMallocHost((void**)&h_pos_embeds, numPosFeats*2*ny*nx*sizeof(T));
    cudaMalloc((void**)&d_pos_embeds, numPosFeats*2*ny*nx*sizeof(T));
    cudaMalloc((void**)&d_dimensions_x, numPosFeats*sizeof(accFloatT));
    cudaMalloc((void**)&d_dimensions_y, numPosFeats*sizeof(accFloatT));

    dim3 blockSize = cuda_config::StandardConfig::block_dim();
    dim3 gridSize((nx+blockSize.x-1)/(blockSize.x), (ny+blockSize.y-1)/(blockSize.y), numPosFeats);

    pos_embedding_kernel<<<gridSize, blockSize>>>(d_pos_embeds, d_dimensions_x, d_dimensions_y, nx, ny);

    cudaMemcpy(h_pos_embeds, d_pos_embeds, numPosFeats*2*ny*nx*sizeof(T), cudaMemcpyDeviceToHost);

    return h_pos_embeds;
}

}