#ifndef NECK_CUH
#define NECK_CUH

#include <cmath>
#include "utils/common.h"
#include "utils/gpu_utils.cuh"
#include "utils/conv/config.cuh"
#include "utils/image_encoder/convolution.cuh"
#include "utils/model.cuh"

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
    Dimensions lower_scale_dims) 
{
    const size_t base_idx = output_channel * lower_scale_dims.y_dimension * lower_scale_dims.x_dimension;
    const size_t idx_y0 = y0 * lower_scale_dims.x_dimension;
    const size_t idx_y1 = y1 * lower_scale_dims.x_dimension;
    
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
                                               Dimensions lower_scale_dims,
                                               Dimensions upper_scale_dims)
{

    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int output_channel = blockIdx.z;

    if (output_channel >= upper_scale_dims.num_channels)
        return;

    if (col <= upper_scale_dims.x_dimension && row <= upper_scale_dims.y_dimension) {
        float origx = static_cast<float>(col)/2;
        float origy = static_cast<float>(row)/2;

        int x0 = static_cast<int>(floor(origx));
        int y0 = static_cast<int>(floor(origy));
        int x1 = min(x0+1, lower_scale_dims.x_dimension-1);
        int y1 = min(y0+1, lower_scale_dims.y_dimension-1);

        float dx = origx - x0;
        float dy = origy - y0;

        accFloatT value = bilinear_interpolate(previous_input, output_channel, x0, x1, y0, y1, dx, dy, lower_scale_dims);

        top_down_feature[output_channel*upper_scale_dims.y_dimension*upper_scale_dims.x_dimension + row*upper_scale_dims.x_dimension + col] = static_cast<T>(value);

        lateral_feature[output_channel*upper_scale_dims.y_dimension*upper_scale_dims.x_dimension + row*upper_scale_dims.x_dimension + col] += static_cast<T>(value);
    }

    accFloatT y_embed = row+1;
    accFloatT x_embed = col+1;

    y_embed = y_embed/(upper_scale_dims.y_dimension + EPSILON) * SCALE;
    x_embed = x_embed/(upper_scale_dims.x_dimension + EPSILON) * SCALE;

    __shared__ accFloatT d_dimensions_x;
    __shared__ accFloatT d_dimensions_y;

    if (col <= upper_scale_dims.x_dimension && row <= upper_scale_dims.y_dimension) {
        if (output_channel < upper_scale_dims.num_channels && threadIdx.x == 0 && threadIdx.y == 0)
        {
            accFloatT power_term = 2*(floorf((output_channel)/2))/upper_scale_dims.num_channels;
            d_dimensions_x = std::pow(TEMPERATURE, power_term);
            d_dimensions_y = std::pow(TEMPERATURE, power_term);
        }

        __syncthreads();

        const bool is_even = output_channel%2 == 0;
        const bool is_first_half = output_channel < upper_scale_dims.num_channels/2;
        const accFloatT embed_val = is_first_half ? y_embed : x_embed;
        const accFloatT dim = is_first_half ? d_dimensions_y : d_dimensions_x;

        accFloatT val = is_even ? std::sin(embed_val / dim) : std::cos(embed_val / dim);
        pos_embeds[output_channel * (upper_scale_dims.y_dimension * upper_scale_dims.x_dimension) + row * upper_scale_dims.x_dimension + col] = static_cast<T>(val);

    }
}

template<typename T, int kernel_size>
void template_conv_and_bilinear_resid_new(x_tensor<T>& x_input, 
                                          x_tensor<T>& x_output,
                                          x_tensor<T>& pos_embeds,
                                          model::NeckLayer<T, model::Nin1, model::Nin2, model::Nin3, model::Nin4>& neck_layer)
{   

    T* d_x_input;
    T* d_x_output;
    T* d_pos_embeds;

    T* d_prev_features = NULL;

    int output_channel = model::Nout;

    int input_channels[4] = {model::Nin1, model::Nin2, model::Nin3, model::Nin4};

    using Config = TileConfig<kernel_size>;

    for (int i = 0; i < neck_layer.size(); i++) {
        T (*weight)[model::Nout][kernel_size][kernel_size] = neck_layer.get_layer_runtime(i)->conv;
        T* bias = neck_layer.get_layer_runtime(i)->bias;

        T* d_weight;
        T* d_bias;

        size_t weight_size = static_cast<size_t>(input_channels[i]) * output_channel * kernel_size * kernel_size;

        gpuErrchk(cudaMalloc((void**)&d_weight, weight_size * sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&d_bias, output_channel * sizeof(T)));

        gpuErrchk(cudaMemcpy(d_weight, weight, weight_size * sizeof(T), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_bias, bias, output_channel * sizeof(T), cudaMemcpyHostToDevice));


        gpuErrchk(cudaMalloc((void**)&d_x_input, x_input.x_dim(i) * x_input.y_dim(i) * x_input.channels(i) * sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&d_x_output, x_output.x_dim(i) * x_output.y_dim(i) * x_output.channels(i) * sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&d_pos_embeds, pos_embeds.x_dim(i) * pos_embeds.y_dim(i) * pos_embeds.channels(i) * sizeof(T)));

        gpuErrchk(cudaMemcpy(d_x_input, x_input.data[i], x_input.x_dim(i) * x_input.y_dim(i) * x_input.channels(i) * sizeof(T), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemset(d_x_output, 0, x_output.x_dim(i) * x_output.y_dim(i) * x_output.channels(i) * sizeof(T)));
        gpuErrchk(cudaMemset(d_pos_embeds, 0, pos_embeds.x_dim(i) * pos_embeds.y_dim(i) * pos_embeds.channels(i) * sizeof(T)));

        dim3 threadsPerBlock(Config::TILE_SIZE, Config::TILE_SIZE, 1);
        dim3 blocksPerGrid((x_input.x_dim(i) + Config::TILE_SIZE - 1) / Config::TILE_SIZE, 
                           (x_input.y_dim(i) + Config::TILE_SIZE - 1) / Config::TILE_SIZE, 
                           x_output.channels(i));

        image_encoder::conv_2d_kernel_direct<T, kernel_size><<<blocksPerGrid, threadsPerBlock>>>(d_x_input, 
                                                                                                d_x_output,
                                                                                                d_weight,
                                                                                                d_bias,
                                                                                                x_input.dims[i],
                                                                                                x_output.dims[i]);

        gpuErrchk(cudaDeviceSynchronize());
        if ((i > 1) && d_prev_features != NULL) {
            T* d_top_down_features;
            gpuErrchk(cudaMalloc((void**)&d_top_down_features, x_output.x_dim(i) * x_output.y_dim(i) * x_output.channels(i) * sizeof(T)));
            gpuErrchk(cudaMemset(d_top_down_features, 0, x_output.x_dim(i) * x_output.y_dim(i) * x_output.channels(i) * sizeof(T)));
            Dimensions lower_scale_dims = {x_output.x_dim(i)/2, x_output.y_dim(i)/2, x_output.channels(i)};

            image_encoder::conv_and_bilinear_resid_kernel<T, kernel_size><<<blocksPerGrid, threadsPerBlock>>>(d_prev_features,
                                                                                                            d_x_output,
                                                                                                            d_top_down_features,
                                                                                                            d_pos_embeds,
                                                                                                            lower_scale_dims,
                                                                                                            x_output.dims[i]);

        } 

        gpuErrchk(cudaDeviceSynchronize());
        d_prev_features = d_x_output;
        
        gpuErrchk(cudaMemcpy(x_output.data[i], d_x_output, x_output.x_dim(i) * x_output.y_dim(i) * x_output.channels(i) * sizeof(T), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(pos_embeds.data[i], d_pos_embeds, pos_embeds.x_dim(i) * pos_embeds.y_dim(i) * pos_embeds.channels(i) * sizeof(T), cudaMemcpyDeviceToHost));
        
        cudaFree(d_weight);
        cudaFree(d_bias);
        cudaFree(d_x_input);
        cudaFree(d_x_output);
        cudaFree(d_pos_embeds);
    }
}

}
#endif
