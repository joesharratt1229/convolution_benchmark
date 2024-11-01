#ifndef NECK_CUH
#define NECK_CUH

#include <cmath>
#include "utils/common.h"
#include "utils/gpu_utils.cuh"
#include "utils/conv/config.cuh"

namespace image_encoder {


template<typename T, int kernel_size>
__global__ void conv_and_bilinear_resid_kernel(T* d_backbone_input,  
                                               T* previous_input,
                                               T* lateral_feature,
                                               T* top_down_feature,
                                               T d_filters[Ni][N1x1][kernel_size][kernel_size],
                                               dims lower_scale_dims,
                                               dims upper_scale_dims)
{
    using Config = TileConfig<kernel_size>;

    unsigned int col = blockIdx.x * Config::TILE_SIZE + threadIdx.x;
    unsigned int row = blockIdx.y * Config::TILE_SIZE + threadIdx.y;
    unsigned int output_channel = blockIdx.z * Config::TILE_SIZE + threadIdx.z;


    __syncthreads();

    T sum1 = 0.0f;

    if (col < upper_scale_dims.width && row < upper_scale_dims.height && output_channel < upper_scale_dims.channel) {
        #pragma unroll
        for (int i = 0; i < lower_scale_dims.channel; i++){
            T filter_val = d_filters[output_channel][i][threadIdx.y][threadIdx.x];
            sum1 += d_backbone_input[i * (upper_scale_dims.width * upper_scale_dims.height) + row*upper_scale_dims.width + col] * filter_val;
        }
        
        lateral_feature[output_channel*upper_scale_dims.height*upper_scale_dims.width + row*upper_scale_dims.width + col] = sum1;
    }


    /*
    for (int y = 2*row; y < 2*row + 1; y++) {
        for (int x = 2*col; x<2*col+1; x++) {
            float origx = x/2;
            float origy = y/2;

            int x0 = static_cast<int>(floor(origx));
            int y0 = static_cast<int>(floor(origy));
            int x1 = min(x0+1, upper_scale_dims.width);
            int y1 = min(y0+1, upper_scale_dims.height);

            float dx = origx - x0;
            float dy = origy - y0;

            float value = (previous_input[output_channel*lower_scale_dims.height*lower_scale_dims.width + y0*lower_scale_dims.width + x0] * (1-dx)*(1-dy) + 
                            previous_input[output_channel*lower_scale_dims.height*lower_scale_dims.width + y0*lower_scale_dims.width + x1] * dx*(1-dy) + 
                            previous_input[output_channel*lower_scale_dims.height*lower_scale_dims.width + y1*lower_scale_dims.width + x0] * (1-dx)*dy + 
                            previous_input[output_channel*lower_scale_dims.height*lower_scale_dims.width + y1*lower_scale_dims.width + x1] * dx*dy);

            top_down_feature[output_channel*upper_scale_dims.height*upper_scale_dims.width + y*upper_scale_dims.width + x] = value;

            lateral_feature[output_channel*upper_scale_dims.height*upper_scale_dims.width + y*upper_scale_dims.width + x] += top_down_feature[output_channel*upper_scale_dims.height*upper_scale_dims.width + y*upper_scale_dims.width + x];
        }
    }*/
}



template<typename T, int kernel_size>
void template_conv_and_bilinear_resid(T* backbone_input,  
                                      T* previous_input,
                                      T* lateral_feature,
                                      T* top_down_feature,
                                      T filters[Ni][N1x1][kernel_size][kernel_size])
{   

    dims lower_scale_dims = {NxPad/2, NyPad/2, Ni};
    dims upper_scale_dims = {NxPad, NyPad, Nn};

    T* d_backbone_input;
    T* d_previous_input;
    T* d_lateral_feature;
    T* d_top_down_feature;
    T (*d_filters)[N1x1][kernel_size][kernel_size];

    using Config = TileConfig<kernel_size>;

    gpuErrchk(cudaMalloc((void**)&d_backbone_input, lower_scale_dims.channel * upper_scale_dims.height * upper_scale_dims.width * sizeof(T)));
    gpuErrchk(cudaMalloc((void**)&d_previous_input, upper_scale_dims.channel* lower_scale_dims.height * lower_scale_dims.width * sizeof(T)));
    gpuErrchk(cudaMalloc((void**)&d_lateral_feature, upper_scale_dims.channel * upper_scale_dims.height * upper_scale_dims.width * sizeof(T)));
    gpuErrchk(cudaMalloc((void**)&d_top_down_feature, upper_scale_dims.channel * upper_scale_dims.height * upper_scale_dims.width * sizeof(T)));
    gpuErrchk(cudaMalloc((void**)&d_filters, Ni * N1x1 * kernel_size * kernel_size * sizeof(T)));

    cudaMemcpy(d_backbone_input, backbone_input, lower_scale_dims.channel * upper_scale_dims.height * upper_scale_dims.width * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_previous_input, previous_input, upper_scale_dims.channel * lower_scale_dims.height * lower_scale_dims.width * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filters, filters, Ni * N1x1 * kernel_size * kernel_size * sizeof(T), cudaMemcpyHostToDevice);



   dim3 threadsPerBlock(Config::TILE_SIZE, Config::TILE_SIZE, 1);
   dim3 blocksPerGrid((upper_scale_dims.width + Config::TILE_SIZE - 1) / Config::TILE_SIZE, (upper_scale_dims.height + Config::TILE_SIZE - 1) / Config::TILE_SIZE, upper_scale_dims.channel);

   conv_and_bilinear_resid_kernel<T, kernel_size><<<blocksPerGrid, threadsPerBlock>>>(d_backbone_input, 
                                                                                   d_previous_input, 
                                                                                   d_lateral_feature, 
                                                                                   d_top_down_feature, 
                                                                                   d_filters,
                                                                                   lower_scale_dims, 
                                                                                   upper_scale_dims);

   gpuErrchk(cudaMemcpy(lateral_feature, d_lateral_feature, upper_scale_dims.channel * upper_scale_dims.height * upper_scale_dims.width * sizeof(T), cudaMemcpyDeviceToHost));
   gpuErrchk(cudaMemcpy(top_down_feature, d_top_down_feature, upper_scale_dims.channel * upper_scale_dims.height * upper_scale_dims.width * sizeof(T), cudaMemcpyDeviceToHost));



}

}
#endif
