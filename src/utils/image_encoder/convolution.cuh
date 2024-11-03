#ifndef CONVOLUTION_CUH
#define CONVOLUTION_CUH

#include "cuda_runtime.h"

#include "utils/conv/config.cuh"
#include "utils/common.h"
#include "utils/gpu_utils.cuh"
#include "utils/conv/config.cuh"

namespace image_encoder {

enum class ConvImplementation {
    Shared,
    Direct    
};

template<typename T, int kernel_size, int in_channel_size, int out_channel_size>
__global__ void conv_2d_kernel_shared(T* d_input, 
                               T* d_output,
                               T d_filters[out_channel_size][in_channel_size][kernel_size][kernel_size],
                               T d_bias[out_channel_size],
                               dims input_dims,
                               dims output_dims)
{
    using Config = TileConfig<kernel_size>;

    unsigned int col = 2*(blockIdx.x * blockDim.x + threadIdx.x);
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int output_channel = blockIdx.z * blockDim.z + threadIdx.z;
    __shared__ T input_cache[in_channel_size][Config::INPUT_TILE_SIZE][Config::INPUT_TILE_SIZE*2];

    if (threadIdx.z < in_channel_size && Config::STRIDE * threadIdx.y < Config::INPUT_TILE_SIZE && Config::STRIDE * threadIdx.x < Config::INPUT_TILE_SIZE) {
        if (threadIdx.y < blockDim.y - 1)
            for (int y = 0; y < Config::STRIDE; y++)
                if (threadIdx.x < blockDim.x - 1)
                    for (int x = 0; x < 2* Config::STRIDE; x++) {
                        int index = threadIdx.z * (input_dims.height * input_dims.width) + 
                                (row * Config::STRIDE + y) * input_dims.width + 
                                (col * Config::STRIDE + x);
                        input_cache[threadIdx.z][Config::STRIDE * threadIdx.y + y][2*Config::STRIDE * threadIdx.x + x] = d_input[index];
                    }
                else
                    for (int x = 0; x < 2* Config::STRIDE + Config::KERNEL_SIZE - 1; x++) {
                        int index = threadIdx.z * (input_dims.height * input_dims.width) + 
                                (row * Config::STRIDE + y) * input_dims.width + 
                                (col * Config::STRIDE + x);
                        input_cache[threadIdx.z][Config::STRIDE * threadIdx.y + y][2*Config::STRIDE * threadIdx.x + x] = d_input[index];
                    }
        else 
            for (int y = 0; y < Config::STRIDE + Config::KERNEL_SIZE - 1; y++)
                if (threadIdx.x < Config::TILE_SIZE - 1)
                    for (int x = 0; x < 2* Config::STRIDE; x++) {
                        int index = threadIdx.z * (input_dims.height * input_dims.width) + 
                                (row * Config::STRIDE + y) * input_dims.width + 
                                (col * Config::STRIDE + x);
                        input_cache[threadIdx.z][Config::STRIDE * threadIdx.y + y][2*Config::STRIDE * threadIdx.x + x] = d_input[index];
                    }
                else
                    for (int x = 0; x < 2* Config::STRIDE + Config::KERNEL_SIZE - 1; x++) {
                        int index = threadIdx.z * (input_dims.height * input_dims.width) + 
                                (row * Config::STRIDE + y) * input_dims.width + 
                                (col * Config::STRIDE + x);
                        input_cache[threadIdx.z][Config::STRIDE * threadIdx.y + y][2*Config::STRIDE * threadIdx.x + x] = d_input[index];
                    }
    }
    
    __syncthreads();

    T sum1 = 0.0f;
    T sum2 = 0.0f;

    if (row < output_dims.height && output_channel < out_channel_size) {
        #pragma unroll
        for (int i = 0; i < in_channel_size; i++)
            #pragma unroll
            for (int y = 0; y < Config::KERNEL_SIZE; y++)
                #pragma unroll
                for (int x = 0; x < Config::KERNEL_SIZE; x++) {
                    T filter_val = d_filters[output_channel][i][y][x];
                    sum1 += input_cache[i][threadIdx.y * Config::STRIDE + y][(2*threadIdx.x) * Config::STRIDE + x] * filter_val;
                    sum2 += input_cache[i][threadIdx.y * Config::STRIDE + y][(2*threadIdx.x+1) * Config::STRIDE + x] * filter_val;

                }
        
        if (col < Ox) {
            d_output[output_channel * output_dims.height * output_dims.width + row * output_dims.width + col] = sum1 + d_bias[output_channel];
        }
        if (col+1 < Ox) {
            d_output[output_channel * output_dims.height * output_dims.width + row * output_dims.width + col + 1] = sum2 + d_bias[output_channel];
        }
    }
}


template<typename T, int kernel_size, int in_channel_size, int out_channel_size>
__global__ void conv_2d_kernel_direct(T* d_input, 
                                     T* d_output,
                                     T d_filters[out_channel_size][in_channel_size][kernel_size][kernel_size],
                                     T d_bias[out_channel_size],
                                     Dimensions input_dims,
                                     Dimensions output_dims)
{
    using Config = TileConfig<kernel_size>;

    unsigned int col = blockIdx.x * Config::TILE_SIZE + threadIdx.x;
    unsigned int row = blockIdx.y * Config::TILE_SIZE + threadIdx.y;
    unsigned int output_channel = blockIdx.z;

    T sum = 0.0f;

    if (col <= output_dims.x_dimension && row <= output_dims.y_dimension && output_channel <= out_channel_size) {
        #pragma unroll
        for (int i = 0; i < in_channel_size; i++){
            #pragma unroll
            for (int y = 0; y < kernel_size; y++)
                #pragma unroll
                for (int x = 0; x < kernel_size; x++) {
                    T filter_val = d_filters[output_channel][i][y][x];
                    sum += d_input[i * (output_dims.x_dimension * output_dims.y_dimension) + (row + y) * output_dims.x_dimension + (col + x)] * filter_val;
                }
        }
        
        d_output[output_channel*output_dims.y_dimension*output_dims.x_dimension + row*output_dims.x_dimension + col] = sum + d_bias[output_channel];
    } 

}


template<typename T, int kernel_size, int in_channel_size, int out_channel_size, ConvImplementation implementation>
__host__ void template_conv_2d(T* h_input, 
                               T* h_output,
                               T h_filters[out_channel_size][in_channel_size][kernel_size][kernel_size],
                               T h_bias[out_channel_size])
{
    using Config = TileConfig<kernel_size>;
    unsigned int Ox2 = (Ox + 1) / 2;

    dim3 threadsPerBlock(Config::TILE_SIZE, Config::TILE_SIZE, Config::CHANNEL_TILE_SIZE);
    dim3 blocksPerGrid((Ox2 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (Oy + threadsPerBlock.y - 1) / threadsPerBlock.y,
                       (Nn + threadsPerBlock.z - 1) / threadsPerBlock.z);

    T* d_input;
    T* d_output;
    T (*d_filters)[in_channel_size][kernel_size][kernel_size];
    T (*d_bias);

    dims input_dims = {NxPad, NyPad, in_channel_size};
    dims output_dims = {Ox, Oy, out_channel_size};

    cudaMalloc((void**)&d_input, input_dims.width * input_dims.height * input_dims.channel * sizeof(T));
    cudaMalloc((void**)&d_output, output_dims.width * output_dims.height * output_dims.channel * sizeof(T));
    cudaMalloc((void**)&d_filters, out_channel_size * in_channel_size * kernel_size * kernel_size * sizeof(T));
    cudaMalloc((void**)&d_bias, out_channel_size * sizeof(T));

    gpuErrchk(cudaMemcpy(d_input, h_input, input_dims.width * input_dims.height * input_dims.channel * sizeof(T), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_filters, h_filters, out_channel_size * in_channel_size * kernel_size * kernel_size * sizeof(T), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_bias, h_bias, out_channel_size * sizeof(T), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    if (implementation == ConvImplementation::Shared)
        conv_2d_kernel_shared<T, kernel_size, in_channel_size, out_channel_size><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_input, d_output, d_filters, d_bias, input_dims, output_dims);
    else
        conv_2d_kernel_direct<T, kernel_size, in_channel_size, out_channel_size><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_input, d_output, d_filters, d_bias, input_dims, output_dims);
    
    
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(h_output, d_output, output_dims.width * output_dims.height * output_dims.channel * sizeof(T), cudaMemcpyDeviceToHost));
}


}



#endif