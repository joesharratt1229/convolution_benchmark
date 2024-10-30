#ifndef NECK_CUH
#define NECK_CUH

#include <cmath>
#include "utils/common.h"
#include "utils/gpu_utils.cuh"

namespace image_encoder {

__constant__ floatT d_filters[Nn][Ni][Ky][Kx];


template<typename T, int tile_size, int input_tile_size, int stride,bool has_previous_input = false>
__global__ void conv_and_bilinear_resid_kernel(T* d_backbone_input,  
                                               T* previous_input,
                                               T* lateral_feature,
                                               T* top_down_feature,
                                               dims lower_scale_dims,
                                               dims upper_scale_dims)
{
    unsigned int col = blockIdx.x * tile_size + threadIdx.x;
    unsigned int row = blockIdx.y * tile_size + threadIdx.y;
    unsigned int output_channel = blockIdx.z * blockDim.z + threadIdx.z;

    __shared__ T input_cache[lower_scale_dims.channel * input_tile_size * input_tile_size];

    if (threadIdx.z < lower_scale_dims.channel && stride * threadIdx.y < input_tile_size && stride * threadIdx.x < input_tile_size) {
        if (threadIdx.y < tile_size - 1)
            for (int y = 0; y < stride; y++)
                if (threadIdx.x < tile_size - 1)
                    for (int x = 0; x < stride; x++) {
                        int index = threadIdx.z * (upper_scale_dims.height * upper_scale_dims.width) + 
                                    (row * stride + y) * upper_scale_dims.width + 
                                    (col * stride + x);

                        input_cache[threadIdx.z * (input_tile_size * input_tile_size) + 
                                    (stride * threadIdx.y + y) * input_tile_size + 
                                    (stride * threadIdx.x + x)] =  d_backbone_input[index];
                    }
                else
                    for (int x = 0; x < stride + Kx - 1; x++) {
                        int index = threadIdx.z * (upper_scale_dims.height * upper_scale_dims.width) + 
                                    (row * stride + y) * upper_scale_dims.width + 
                                    (col * stride + x);

                        input_cache[threadIdx.z * (input_tile_size * input_tile_size) + 
                                    (stride * threadIdx.y + y) * input_tile_size + 
                                    (stride * threadIdx.x + x)] = d_backbone_input[index];
                    }
        else 
            for (int y = 0; y < stride + Ky - 1; y++)
                if (threadIdx.x < tile_size - 1)
                    for (int x = 0; x < stride; x++) {
                        int index = threadIdx.z * (upper_scale_dims.height * upper_scale_dims.width) + 
                                 (row * stride + y) * upper_scale_dims.width + 
                                 (col * stride + x);

                        input_cache[threadIdx.z * (input_tile_size * input_tile_size) + 
                                    (stride * threadIdx.y + y) * input_tile_size + 
                                    (stride * threadIdx.x + x)] = d_backbone_input[index];
                    }
                else
                    for (int x = 0; x < stride + Kx - 1; x++) {
                        int index = threadIdx.z * (upper_scale_dims.height * upper_scale_dims.width) +    
                                 (row * stride + y) * upper_scale_dims.width + 
                                 (col * stride + x);

                        input_cache[threadIdx.z * (input_tile_size * input_tile_size) + 
                                    (stride * threadIdx.y + y) * input_tile_size + 
                                    (stride * threadIdx.x + x)] = d_backbone_input[index];
                    }
    }

    __syncthreads();

    T sum1 = 0.0f;

    if (col < upper_scale_dims.width && row < upper_scale_dims.height && output_channel < upper_scale_dims.channel) {
        #pragma unroll
        for (int i = 0; i < lower_scale_dims.channel; i++)
            #pragma unroll
            for (int y = 0; y < Ky; y++)
                #pragma unroll
                for (int x = 0; x < Kx; x++) {
                    T filter_val = d_filters[output_channel][i][y][x];
                    sum1 += input_cache[i * (input_tile_size * input_tile_size) + 
                                        (stride * threadIdx.y + y) * input_tile_size + 
                                        (stride * threadIdx.x + x)] * filter_val;
                }
        
        
        lateral_feature[output_channel*upper_scale_dims.height*upper_scale_dims.width + row*upper_scale_dims.width + col] = sum1;
    }

    __syncthreads();



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
    }
}

void template_conv_and_bilinear_resid(floatT* backbone_input,  
                                      floatT* previous_input,
                                      floatT* lateral_feature,
                                      floatT* top_down_feature,
                                      floatT* filters,
                                      dims lower_scale_dims,
                                      dims upper_scale_dims)
{   
    floatT* d_backbone_input;
    floatT* d_previous_input;
    floatT* d_lateral_feature;
    floatT* d_top_down_feature;

    gpuErrchk(cudaMalloc((void**)&d_backbone_input, lower_scale_dims.channel * upper_scale_dims.height * upper_scale_dims.width * sizeof(floatT)));
    gpuErrchk(cudaMalloc((void**)&d_previous_input, upper_scale_dims.channel* lower_scale_dims.height * lower_scale_dims.width * sizeof(floatT)));
    gpuErrchk(cudaMalloc((void**)&d_lateral_feature, upper_scale_dims.channel * upper_scale_dims.height * upper_scale_dims.width * sizeof(floatT)));
    gpuErrchk(cudaMalloc((void**)&d_top_down_feature, upper_scale_dims.channel * upper_scale_dims.height * upper_scale_dims.width * sizeof(floatT)));


    cudaMemcpy(d_backbone_input, backbone_input, lower_scale_dims.channel * upper_scale_dims.height * upper_scale_dims.width * sizeof(floatT), cudaMemcpyHostToDevice);
    cudaMemcpy(d_previous_input, previous_input, upper_scale_dims.channel * lower_scale_dims.height * lower_scale_dims.width * sizeof(floatT), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_filters, filters, Nn * Ni * Ky * Kx * sizeof(floatT), 0, cudaMemcpyHostToDevice);

    int tile_size = 32;
    int input_tile_size = 2*tile_size + Ky - 1;
    int stride = 1;

   dim



}

}
#endif
