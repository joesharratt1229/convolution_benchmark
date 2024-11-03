#ifndef NECK_CUH
#define NECK_CUH

#include <cmath>
#include "utils/common.h"
#include "utils/gpu_utils.cuh"
#include "utils/conv/config.cuh"
#include "utils/image_encoder/convolution.cuh"

namespace image_encoder {

#define EPSILON 1e-6
#define SCALE 2*M_PI


template<typename T>
__device__ __forceinline__ accFloatT bilinear_interpolate(
    const T* input,
    int output_channel,
    int x0, int x1,
    int y0, int y1,
    float dx, float dy,
    dims lower_scale_dims) 
{
    const size_t base_idx = output_channel * lower_scale_dims.height * lower_scale_dims.width;
    const size_t idx_y0 = y0 * lower_scale_dims.width;
    const size_t idx_y1 = y1 * lower_scale_dims.width;
    
    return ((1-dx)*(1-dy) * static_cast<accFloatT>(input[base_idx + idx_y0 + x0]) + 
            dx*(1-dy) * static_cast<accFloatT>(input[base_idx + idx_y0 + x1]) + 
            (1-dx)*dy * static_cast<accFloatT>(input[base_idx + idx_y1 + x0]) + 
            dx*dy * static_cast<accFloatT>(input[base_idx + idx_y1 + x1]));
}

template<typename T, int kernel_size>
__global__ void conv_and_bilinear_resid_kernel(T* previous_input,
                                               T* lateral_feature,
                                               T* top_down_feature,
                                               T* pos_embeds,
                                               dims lower_scale_dims,
                                               dims upper_scale_dims)
{

    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int output_channel = blockIdx.z;

    if (output_channel >= upper_scale_dims.channel)
        return;

    if (col <= upper_scale_dims.width && row <= upper_scale_dims.height) {
        float origx = static_cast<float>(col)/2;
        float origy = static_cast<float>(row)/2;

        int x0 = static_cast<int>(floor(origx));
        int y0 = static_cast<int>(floor(origy));
        int x1 = min(x0+1, lower_scale_dims.width-1);
        int y1 = min(y0+1, lower_scale_dims.height-1);

        float dx = origx - x0;
        float dy = origy - y0;

        accFloatT value = bilinear_interpolate(previous_input, output_channel, x0, x1, y0, y1, dx, dy, lower_scale_dims);

        top_down_feature[output_channel*upper_scale_dims.height*upper_scale_dims.width + row*upper_scale_dims.width + col] = static_cast<T>(value);

        lateral_feature[output_channel*upper_scale_dims.height*upper_scale_dims.width + row*upper_scale_dims.width + col] += static_cast<T>(value);
    }

    accFloatT y_embed = row+1;
    accFloatT x_embed = col+1;

    y_embed = y_embed/(upper_scale_dims.height + EPSILON) * SCALE;
    x_embed = x_embed/(upper_scale_dims.width + EPSILON) * SCALE;

    __shared__ accFloatT d_dimensions_x;
    __shared__ accFloatT d_dimensions_y;

    if (col <= upper_scale_dims.width && row <= upper_scale_dims.height) {
        if (output_channel < upper_scale_dims.channel && threadIdx.x == 0 && threadIdx.y == 0)
        {
            accFloatT power_term = 2*(floorf((output_channel)/2))/upper_scale_dims.channel;
            d_dimensions_x = std::pow(TEMPERATURE, power_term);
            d_dimensions_y = std::pow(TEMPERATURE, power_term);
        }

        __syncthreads();

        const bool is_even = output_channel%2 == 0;
        const bool is_first_half = output_channel < upper_scale_dims.channel/2;
        const accFloatT embed_val = is_first_half ? y_embed : x_embed;
        const accFloatT dim = is_first_half ? d_dimensions_y : d_dimensions_x;

        accFloatT val = is_even ? std::sin(embed_val / dim) : std::cos(embed_val / dim);
        pos_embeds[output_channel * (upper_scale_dims.height * upper_scale_dims.width) + row * upper_scale_dims.width + col] = static_cast<T>(val);

    }
}


template<typename T, int kernel_size>
void template_conv_and_bilinear_resid(T* backbone_input,  
                                      T* previous_input,
                                      T* lateral_feature,
                                      T* top_down_feature,
                                      T filters[Nn][N1x1][kernel_size][kernel_size],
                                      T bias[Nn])
{   

    dims lower_scale_dims = {Nx/2, Ny/2, N1x1};
    dims upper_scale_dims = {Nx, Ny, Nn};

    T* h_pos_embeds;

    gpuErrchk(cudaMallocHost((void**)&h_pos_embeds, numPosFeats*upper_scale_dims.height*upper_scale_dims.width*sizeof(T)));

    T* d_backbone_input, *d_previous_input, *d_lateral_feature, *d_top_down_feature, *d_pos_embeds; 
    T (*d_filters)[N1x1][kernel_size][kernel_size];
    T* d_bias;
    using Config = TileConfig<kernel_size>;

    gpuErrchk(cudaMalloc((void**)&d_backbone_input, lower_scale_dims.channel * upper_scale_dims.height * upper_scale_dims.width * sizeof(T)));
    gpuErrchk(cudaMalloc((void**)&d_previous_input, upper_scale_dims.channel* lower_scale_dims.height * lower_scale_dims.width * sizeof(T)));
    gpuErrchk(cudaMalloc((void**)&d_lateral_feature, upper_scale_dims.channel * upper_scale_dims.height * upper_scale_dims.width * sizeof(T)));
    gpuErrchk(cudaMalloc((void**)&d_top_down_feature, upper_scale_dims.channel * upper_scale_dims.height * upper_scale_dims.width * sizeof(T)));

    gpuErrchk(cudaMalloc((void**)&d_pos_embeds, numPosFeats*upper_scale_dims.height*upper_scale_dims.width*sizeof(T)));
    gpuErrchk(cudaMalloc((void**)&d_filters, Nn * N1x1 * kernel_size * kernel_size * sizeof(T)));
    gpuErrchk(cudaMalloc((void**)&d_bias, Nn * sizeof(T)));

    gpuErrchk(cudaMemcpy(d_backbone_input, backbone_input, lower_scale_dims.channel * upper_scale_dims.height * upper_scale_dims.width * sizeof(T), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_previous_input, previous_input, upper_scale_dims.channel * lower_scale_dims.height * lower_scale_dims.width * sizeof(T), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_filters, filters, Nn * N1x1 * kernel_size * kernel_size * sizeof(T), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_bias, bias, Nn * sizeof(T), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemset(d_pos_embeds, 0, numPosFeats*upper_scale_dims.height*upper_scale_dims.width*sizeof(T)));
    gpuErrchk(cudaMemset(d_lateral_feature, 0, upper_scale_dims.channel * upper_scale_dims.height * upper_scale_dims.width * sizeof(T)));

   dim3 threadsPerBlock(Config::TILE_SIZE, Config::TILE_SIZE, 1);
   dim3 blocksPerGrid((upper_scale_dims.width + Config::TILE_SIZE - 1) / Config::TILE_SIZE, 
                      (upper_scale_dims.height + Config::TILE_SIZE - 1) / Config::TILE_SIZE, 
                      upper_scale_dims.channel);

    image_encoder::conv_2d_kernel_direct<T, kernel_size, N1x1, Nn><<<blocksPerGrid, threadsPerBlock>>>(d_backbone_input, 
                                                                                                       d_lateral_feature, 
                                                                                                       d_filters, 
                                                                                                       d_bias,
                                                                                                       lower_scale_dims, 
                                                                                                       upper_scale_dims);

   conv_and_bilinear_resid_kernel<T, kernel_size><<<blocksPerGrid, threadsPerBlock>>>(d_previous_input, 
                                                                                      d_lateral_feature, 
                                                                                      d_top_down_feature, 
                                                                                      d_pos_embeds,
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

}



template<typename T, int kernel_size>
void template_conv_and_bilinear_resid_new(x_tensor<T>& x_input, 
                                          x_tensor<T>& x_output,
                                          x_tensor<T>& pos_embeds,
                                          model::NeckLayer<floatT, model::Nin1, model::Nin2, model::Nin3, model::Nin4>* neck_layer)
{   

    x_tensor<T> d_x_input;
    x_tensor<T> d_x_output;
    x_tensor<T> d_pos_embeds;


    for (int i = 0; i < neck_layer.size(); i++) {

        gpuErrchk(cudaMalloc((void**)&d_x_input.data[i], x_input.x_dim(i) * x_input.y_dim(i) * x_input.channels(i) * sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&d_x_output.data[i], x_output.x_dim(i) * x_output.y_dim(i) * x_output.channels(i) * sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&d_pos_embeds.data[i], pos_embeds.x_dim(i) * pos_embeds.y_dim(i) * pos_embeds.channels(i) * sizeof(T)));

        gpuErrchk(cudaMemcpy(d_x_input.data[i], x_input.data[i], x_input.x_dim(i) * x_input.y_dim(i) * x_input.channels(i) * sizeof(T), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemset(d_x_output.data[i], 0, x_output.x_dim(i) * x_output.y_dim(i) * x_output.channels(i) * sizeof(T)));
        gpuErrchk(cudaMemset(d_pos_embeds.data[i], 0, pos_embeds.x_dim(i) * pos_embeds.y_dim(i) * pos_embeds.channels(i) * sizeof(T)));

        if (i == 0 || i == 1) {
            dim3 threadsPerBlock(Config::TILE_SIZE, Config::TILE_SIZE, 1);
            dim3 blocksPerGrid((x_input.x_dim(i) + Config::TILE_SIZE - 1) / Config::TILE_SIZE, 
                               (x_input.y_dim(i) + Config::TILE_SIZE - 1) / Config::TILE_SIZE, 
                               x_input.channels(i));

            auto* weight = neck_layer->get_layer_runtime(i)->conv;
            auto* bias = neck_layer->get_layer_runtime(i)->bias;

            image_encoder::conv_2d_kernel_direct<T, kernel_size, model::Nin1, model::Nin2><<<blocksPerGrid, threadsPerBlock>>>(d_x_input.data[i], 
                                                                                                                               d_x_output.data[i],
                                                                                                                               
                                                                                                                               x_input.dims[i],
                                                                                                                               x_output.dims[i]);

            gpuErrchk(cudaDeviceSynchronize());
            cudaMemcpy(x_output.data[i], d_x_output.data[i], x_output.x_dim(i) * x_output.y_dim(i) * x_output.channels(i) * sizeof(T), cudaMemcpyDeviceToHost);
        }
    }

    cudaFree(d_x_input.data);
    cudaFree(d_x_output.data);
    cudaFree(d_pos_embeds.data);


}

}
#endif
