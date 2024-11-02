#ifndef NECK_CUH
#define NECK_CUH

#include <cmath>
#include "utils/common.h"
#include "utils/gpu_utils.cuh"
#include "utils/conv/config.cuh"

namespace image_encoder {


__constant__ floatT d_filters[Nn][N1x1][TileConfig<1>::KERNEL_SIZE][TileConfig<1>::KERNEL_SIZE];

template<typename T, int kernel_size>
__global__ void conv_and_bilinear_resid_kernel(T* d_backbone_input,  
                                               T* previous_input,
                                               T* lateral_feature,
                                               T* top_down_feature,
                                               T* pos_embeds,
                                               accFloatT* d_dimensions_x,
                                               accFloatT* d_dimensions_y,
                                               dims lower_scale_dims,
                                               dims upper_scale_dims)
{
    using Config = TileConfig<kernel_size>;

    unsigned int col = blockIdx.x * Config::TILE_SIZE + threadIdx.x;
    unsigned int row = blockIdx.y * Config::TILE_SIZE + threadIdx.y;
    unsigned int output_channel = blockIdx.z;

    T sum1 = 0.0f;

    if (col <= upper_scale_dims.width && row <= upper_scale_dims.height && output_channel <= upper_scale_dims.channel) {
        #pragma unroll
        for (int i = 0; i < lower_scale_dims.channel; i++){
            T filter_val = d_filters[output_channel][i][0][0];
            sum1 += d_backbone_input[i * (upper_scale_dims.width * upper_scale_dims.height) + (row)*upper_scale_dims.width + (col)] * filter_val;
        }
        
        lateral_feature[output_channel*upper_scale_dims.height*upper_scale_dims.width + row*upper_scale_dims.width + col] = sum1;
    } 

    __syncthreads();  

    if (col <= upper_scale_dims.width && row <= upper_scale_dims.height && output_channel <= upper_scale_dims.channel) {
        float origx = static_cast<float>(col)/2;
        float origy = static_cast<float>(row)/2;

        int x0 = static_cast<int>(floor(origx));
        int y0 = static_cast<int>(floor(origy));
        int x1 = min(x0+1, lower_scale_dims.width-1);
        int y1 = min(y0+1, lower_scale_dims.height-1);

        float dx = origx - x0;
        float dy = origy - y0;

        accFloatT value = ((1-dx)*(1-dy) * static_cast<accFloatT>(previous_input[output_channel*lower_scale_dims.height*lower_scale_dims.width + y0*lower_scale_dims.width + x0]) + 
                   dx*(1-dy) * static_cast<accFloatT>(previous_input[output_channel*lower_scale_dims.height*lower_scale_dims.width + y0*lower_scale_dims.width + x1]) + 
                   (1-dx)*dy * static_cast<accFloatT>(previous_input[output_channel*lower_scale_dims.height*lower_scale_dims.width + y1*lower_scale_dims.width + x0]) + 
                   dx*dy * static_cast<accFloatT>(previous_input[output_channel*lower_scale_dims.height*lower_scale_dims.width + y1*lower_scale_dims.width + x1]));

        top_down_feature[output_channel*upper_scale_dims.height*upper_scale_dims.width + row*upper_scale_dims.width + col] = static_cast<T>(value);

        lateral_feature[output_channel*upper_scale_dims.height*upper_scale_dims.width + row*upper_scale_dims.width + col] += static_cast<T>(value);
    }

    accFloatT y_embed = row+1;
    accFloatT x_embed = col+1;

    if (col <= upper_scale_dims.width && row <= upper_scale_dims.height) {
        if (output_channel < numPosFeats && col == 0 && row == 0)
        {
            accFloatT power_term = 2*(floorf((output_channel)/2))/numPosFeats;
            d_dimensions_x[output_channel] = std::pow(TEMPERATURE, power_term);
            d_dimensions_y[output_channel] = std::pow(TEMPERATURE, power_term);
        }

        __syncthreads();

        if (output_channel < numPosFeats)
        {
            const bool is_even = output_channel%2 == 0;
            const bool is_first_half = output_channel < numPosFeats/2;
            const accFloatT embed_val = is_first_half ? y_embed : x_embed;
            const accFloatT dim = is_first_half ? d_dimensions_y[output_channel] : d_dimensions_x[output_channel];

            accFloatT val = is_even ? std::sin(embed_val / dim) : std::cos(embed_val / dim);
            pos_embeds[output_channel * (upper_scale_dims.height * upper_scale_dims.width) + row * upper_scale_dims.width + col] = static_cast<T>(val);
        }
    }
}


template<typename T, int kernel_size>
void template_conv_and_bilinear_resid(T* backbone_input,  
                                      T* previous_input,
                                      T* lateral_feature,
                                      T* top_down_feature,
                                      T filters[Nn][N1x1][kernel_size][kernel_size])
{   

    dims lower_scale_dims = {Nx/2, Ny/2, N1x1};
    dims upper_scale_dims = {Nx, Ny, Nn};

    T* h_pos_embeds;

    gpuErrchk(cudaMallocHost((void**)&h_pos_embeds, numPosFeats*upper_scale_dims.height*upper_scale_dims.width*sizeof(T)));

    T* d_backbone_input;
    T* d_previous_input;
    T* d_lateral_feature;
    T* d_top_down_feature;
    T* d_pos_embeds;
    accFloatT* d_dimensions_x;
    accFloatT* d_dimensions_y;

    using Config = TileConfig<kernel_size>;

    gpuErrchk(cudaMalloc((void**)&d_backbone_input, lower_scale_dims.channel * upper_scale_dims.height * upper_scale_dims.width * sizeof(T)));
    gpuErrchk(cudaMalloc((void**)&d_previous_input, upper_scale_dims.channel* lower_scale_dims.height * lower_scale_dims.width * sizeof(T)));
    gpuErrchk(cudaMalloc((void**)&d_lateral_feature, upper_scale_dims.channel * upper_scale_dims.height * upper_scale_dims.width * sizeof(T)));
    gpuErrchk(cudaMalloc((void**)&d_top_down_feature, upper_scale_dims.channel * upper_scale_dims.height * upper_scale_dims.width * sizeof(T)));

    gpuErrchk(cudaMalloc((void**)&d_pos_embeds, numPosFeats*upper_scale_dims.height*upper_scale_dims.width*sizeof(T)));
    gpuErrchk(cudaMalloc((void**)&d_dimensions_x, numPosFeats*sizeof(accFloatT)));
    gpuErrchk(cudaMalloc((void**)&d_dimensions_y, numPosFeats*sizeof(accFloatT)));

    gpuErrchk(cudaMemcpy(d_backbone_input, backbone_input, lower_scale_dims.channel * upper_scale_dims.height * upper_scale_dims.width * sizeof(T), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_previous_input, previous_input, upper_scale_dims.channel * lower_scale_dims.height * lower_scale_dims.width * sizeof(T), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(d_filters, filters, Nn * N1x1 * kernel_size * kernel_size * sizeof(T)));

    gpuErrchk(cudaMemset(d_pos_embeds, 0, numPosFeats*upper_scale_dims.height*upper_scale_dims.width*sizeof(T)));
    gpuErrchk(cudaMemset(d_dimensions_x, 0, numPosFeats*sizeof(accFloatT)));
    gpuErrchk(cudaMemset(d_dimensions_y, 0, numPosFeats*sizeof(accFloatT)));



   dim3 threadsPerBlock(Config::TILE_SIZE, Config::TILE_SIZE, 1);
   dim3 blocksPerGrid((upper_scale_dims.width + Config::TILE_SIZE - 1) / Config::TILE_SIZE, 
                      (upper_scale_dims.height + Config::TILE_SIZE - 1) / Config::TILE_SIZE, 
                      upper_scale_dims.channel);

   conv_and_bilinear_resid_kernel<T, kernel_size><<<blocksPerGrid, threadsPerBlock>>>(d_backbone_input, 
                                                                                    d_previous_input, 
                                                                                    d_lateral_feature, 
                                                                                    d_top_down_feature, 
                                                                                    d_pos_embeds,
                                                                                    d_dimensions_x,
                                                                                    d_dimensions_y,
                                                                                    lower_scale_dims, 
                                                                                    upper_scale_dims);

   gpuErrchk(cudaMemcpy(lateral_feature, d_lateral_feature, upper_scale_dims.channel * upper_scale_dims.height * upper_scale_dims.width * sizeof(T), cudaMemcpyDeviceToHost));
   gpuErrchk(cudaMemcpy(top_down_feature, d_top_down_feature, upper_scale_dims.channel * upper_scale_dims.height * upper_scale_dims.width * sizeof(T), cudaMemcpyDeviceToHost));
   gpuErrchk(cudaMemcpy(h_pos_embeds, d_pos_embeds, numPosFeats*upper_scale_dims.height*upper_scale_dims.width*sizeof(T), cudaMemcpyDeviceToHost));

   cudaFree(d_backbone_input);
   cudaFree(d_previous_input);
   cudaFree(d_lateral_feature);
   cudaFree(d_top_down_feature);
   cudaFree(d_pos_embeds);
   cudaFree(d_dimensions_x);
   cudaFree(d_dimensions_y);

}

}
#endif
