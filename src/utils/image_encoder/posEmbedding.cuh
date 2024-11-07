#include <cmath>
#include "cuda_runtime.h"

#include "utils/common.h"
#include "utils/gpu_utils.cuh"


namespace image_encoder {       

template<typename T, typename accFloatT>
__global__ void pos_embedding_kernel(T* pos_embeds,
                                    Dimensions output_dims)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y ;
    int pos_feat = threadIdx.z + blockIdx.z * blockDim.z;

    if (pos_feat >= output_dims.num_channels || x >= output_dims.x_dimension || y >= output_dims.y_dimension) return;

    __shared__ accFloatT d_dimensions_x;
    __shared__ accFloatT d_dimensions_y;

    accFloatT y_embed = y+1;
    accFloatT x_embed = x+1;

    y_embed = y_embed/(output_dims.y_dimension + EPSILON) * SCALE;
    x_embed = x_embed/(output_dims.x_dimension + EPSILON) * SCALE;

    if (pos_feat < output_dims.num_channels && x == 0 && y == 0)
    {
        accFloatT power_term = 2*(floorf((pos_feat)/2))/output_dims.num_channels;
        d_dimensions_x = std::pow(TEMPERATURE, power_term);
        d_dimensions_y = std::pow(TEMPERATURE, power_term);
    }

    __syncthreads();

    if (pos_feat < output_dims.num_channels && x < output_dims.x_dimension && y < output_dims.y_dimension)
    {
       const bool is_even = pos_feat%2 == 0;
       const bool is_first_half = pos_feat < output_dims.num_channels/2;
       const accFloatT embed_val = is_first_half ? y_embed : x_embed;
       const accFloatT dim = is_first_half ? d_dimensions_y : d_dimensions_x;

       accFloatT val = is_even ? std::sin(embed_val / dim) : std::cos(embed_val / dim);
       pos_embeds[pos_feat * (output_dims.y_dimension * output_dims.x_dimension) + y * output_dims.x_dimension + x] = static_cast<T>(val);
    }
}



template<typename T, typename accFloatT>
T* template_pos_embedding(Dimensions output_dims)
{
    T* h_pos_embeds;
    T* d_pos_embeds;
    accFloatT* d_dimensions_x;
    accFloatT* d_dimensions_y;

    cudaMallocHost((void**)&h_pos_embeds, output_dims.num_channels*2*output_dims.y_dimension*output_dims.x_dimension*sizeof(T));
    cudaMalloc((void**)&d_pos_embeds, output_dims.num_channels*2*output_dims.y_dimension*output_dims.x_dimension*sizeof(T));
    cudaMalloc((void**)&d_dimensions_x, output_dims.num_channels*sizeof(accFloatT));
    cudaMalloc((void**)&d_dimensions_y, output_dims.num_channels*sizeof(accFloatT));

    dim3 blockSize = cuda_config::StandardConfig::block_dim();
    dim3 gridSize((output_dims.x_dimension+blockSize.x-1)/(blockSize.x), (output_dims.y_dimension+blockSize.y-1)/(blockSize.y), output_dims.num_channels);

    pos_embedding_kernel<<<gridSize, blockSize>>>(d_pos_embeds, output_dims);

    cudaMemcpy(h_pos_embeds, d_pos_embeds, output_dims.num_channels*2*output_dims.y_dimension*output_dims.x_dimension*sizeof(T), cudaMemcpyDeviceToHost);

    return h_pos_embeds;
}

}