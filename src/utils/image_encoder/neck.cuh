#ifndef NECK_CUH
#define NECK_CUH

#include <cmath>
#include "utils/common.h"
#include "utils/gpu_utils.cuh"
#include "utils/conv/config.cuh"
#include "utils/image_encoder/convolution.cuh"
#include "utils/model.cuh"

namespace image_encoder {


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
                                               T* pos_embeds,
                                               Dimensions lower_scale_dims,
                                               Dimensions upper_scale_dims)
{

    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int output_channel = blockIdx.z;

    if (output_channel >= upper_scale_dims.num_channels || col >= upper_scale_dims.x_dimension || row >= upper_scale_dims.y_dimension)
        return;

    if (col < upper_scale_dims.x_dimension && row < upper_scale_dims.y_dimension) {
        float origx = static_cast<float>(col)/2;
        float origy = static_cast<float>(row)/2;

        int x0 = static_cast<int>(floor(origx));
        int y0 = static_cast<int>(floor(origy));
        int x1 = min(x0+1, lower_scale_dims.x_dimension-1);
        int y1 = min(y0+1, lower_scale_dims.y_dimension-1);

        float dx = origx - x0;
        float dy = origy - y0;

        accFloatT value = bilinear_interpolate(previous_input, output_channel, x0, x1, y0, y1, dx, dy, lower_scale_dims);

        lateral_feature[output_channel*upper_scale_dims.y_dimension*upper_scale_dims.x_dimension + row*upper_scale_dims.x_dimension + col] += static_cast<T>(value);
    }

    accFloatT y_embed = row+1;
    accFloatT x_embed = col+1;

    y_embed = y_embed/(upper_scale_dims.y_dimension + EPSILON) * SCALE;
    x_embed = x_embed/(upper_scale_dims.x_dimension + EPSILON) * SCALE;

    __shared__ accFloatT d_dimensions_x;
    __shared__ accFloatT d_dimensions_y;


    if (col < upper_scale_dims.x_dimension && row < upper_scale_dims.y_dimension && output_channel < upper_scale_dims.num_channels) {
        if (threadIdx.x == 0 && threadIdx.y == 0)
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
void template_conv_and_bilinear_resid_new(XTensor<T>** x_input, 
                                          XTensor<T>** x_output,
                                          XTensor<T>** pos_embeds,
                                          model::NeckLayer<T, model::Nin1, model::Nin2, model::Nin3, model::Nin4>& neck_layer)
{   

    XTensor<T>** d_x_input = new XTensor<T>*[4];
    XTensor<T>** d_x_output = new XTensor<T>*[4];
    XTensor<T>** d_pos_embeds = new XTensor<T>*[4];

    cudaStream_t streams[5];

    int output_channel = model::Nout;
    int input_channels[4] = {model::Nin1, model::Nin2, model::Nin3, model::Nin4};

    for (int i = 0; i < neck_layer.size(); i++) {
        gpuErrchk(cudaStreamCreate(&streams[i]));
        d_x_input[i] = new XTensor<T>(*x_input[i], x_input[i]->get_dims(), streams[i], true, true);
        d_x_output[i] = new XTensor<T>(*x_output[i], x_output[i]->get_dims(), streams[i], true, true);
        d_pos_embeds[i] = new XTensor<T>(*pos_embeds[i], pos_embeds[i]->get_dims(), streams[i], true, true);
    }

    using Config = TileConfig<kernel_size>;

    for (int i = 0; i < neck_layer.size(); i++) {
        T (*weight)[model::Nout][kernel_size][kernel_size] = neck_layer.get_layer_runtime(i)->conv;
        T* bias = neck_layer.get_layer_runtime(i)->bias;

        T* d_weight;
        T* d_bias;

        dim3 threadsPerBlock(Config::TILE_SIZE, Config::TILE_SIZE, 1);
        dim3 blocksPerGrid((x_output[i]->x_dim() + Config::TILE_SIZE - 1) / Config::TILE_SIZE, 
                            (x_output[i]->y_dim() + Config::TILE_SIZE - 1) / Config::TILE_SIZE, 
                            x_output[i]->channels());

        size_t weight_size = static_cast<size_t>(input_channels[i]) * output_channel * kernel_size * kernel_size;

        gpuErrchk(cudaMalloc((void**)&d_weight, weight_size * sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&d_bias, output_channel * sizeof(T)));

        gpuErrchk(cudaMemcpyAsync(d_weight, weight, weight_size * sizeof(T), cudaMemcpyHostToDevice, 0));
        gpuErrchk(cudaMemcpyAsync(d_bias, bias, output_channel * sizeof(T), cudaMemcpyHostToDevice, 0));

        image_encoder::conv_2d_kernel_direct<T, kernel_size><<<blocksPerGrid, threadsPerBlock, 0, 0>>>(d_x_input[i]->get(), 
                                                                                                       d_x_output[i]->get(),
                                                                                                       d_weight,
                                                                                                       d_bias,
                                                                                                       x_input[i]->get_dims(),
                                                                                                       x_output[i]->get_dims());;

        if ((i > 1)) {
            image_encoder::conv_and_bilinear_resid_kernel<T, kernel_size><<<blocksPerGrid, threadsPerBlock, 0, 0>>>(d_x_output[i-1]->get(),
                                                                                                                               d_x_output[i]->get(),
                                                                                                                               d_pos_embeds[i]->get(),
                                                                                                                               x_output[i-1]->get_dims(),
                                                                                                                               x_output[i]->get_dims());
        } else {
            image_encoder::pos_embedding_kernel<T, accFloatT><<<blocksPerGrid, threadsPerBlock, 0, 0>>>(d_pos_embeds[i]->get(),
                                                                                                    d_pos_embeds[i]->get_dims());
        }

        gpuErrchk(cudaFree(d_weight));
        gpuErrchk(cudaFree(d_bias));
    }

    
    for (int i = 0; i < neck_layer.size(); i++) {
        size_t output_size = x_output[i]->x_dim() * x_output[i]->y_dim() * x_output[i]->channels() * sizeof(T);
        gpuErrchk(cudaMemcpyAsync(x_output[i]->get(), d_x_output[i]->get(), output_size, cudaMemcpyDeviceToHost, streams[i]));
        gpuErrchk(cudaMemcpyAsync(pos_embeds[i]->get(), d_pos_embeds[i]->get(), output_size, cudaMemcpyDeviceToHost, streams[i]));
    }


}
}
#endif
