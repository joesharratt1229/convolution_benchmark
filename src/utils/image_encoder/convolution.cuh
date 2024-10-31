#ifndef CONVOLUTION_CUH
#define CONVOLUTION_CUH

#include "cuda_runtime.h"

#include "utils/conv/config.cuh"
#include "utils/common.h"
#include "utils/gpu_utils.cuh"

namespace image_encoder {   

template<typename T, int kernel_size>
__global__ void conv_2d_kernel(T* d_input, 
                               T* d_output,
                               T d_filters[Nn][Ni][Ky][Kx],
                               dims input_dims,
                               dims output_dims)
{
    using Config = TileConfig<kernel_size>;

    unsigned int col = 2*(blockIdx.x * blockDim.x + threadIdx.x);
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int output_channel = blockIdx.z * blockDim.z + threadIdx.z;
    __shared__ T input_cache[Ni][Config::INPUT_TILE_SIZE][Config::INPUT_TILE_SIZE*2];

    if (threadIdx.z < input_dims.channel && Config::STRIDE * threadIdx.y < Config::INPUT_TILE_SIZE && Config::STRIDE * threadIdx.x < Config::INPUT_TILE_SIZE) {
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

    if (row < output_dims.height && output_channel < Nn) {
        #pragma unroll
        for (int i = 0; i < Ni; i++)
            #pragma unroll
            for (int y = 0; y < Config::KERNEL_SIZE; y++)
                #pragma unroll
                for (int x = 0; x < Config::KERNEL_SIZE; x++) {
                    T filter_val = d_filters[output_channel][i][y][x];
                    sum1 += input_cache[i][threadIdx.y * Config::STRIDE + y][(2*threadIdx.x) * Config::STRIDE + x] * filter_val;
                    sum2 += input_cache[i][threadIdx.y * Config::STRIDE + y][(2*threadIdx.x+1) * Config::STRIDE + x] * filter_val;

                }
        
        if (col < Ox) {
            d_output[output_channel * output_dims.height * output_dims.width + row * output_dims.width + col] = sum1;
        }
        if (col+1 < Ox) {
            d_output[output_channel * output_dims.height * output_dims.width + row * output_dims.width + col + 1] = sum2;
        }
    }
}



template<typename T, int kernel_size>
__host__ void template_conv_2d(T* h_input, 
                               T* h_output,
                               T h_filters[Nn][Ni][Ky][Kx])
{
    using Config = TileConfig<kernel_size>;
    unsigned int Ox2 = (Ox + 1) / 2;

    dim3 threadsPerBlock(Config::TILE_SIZE, Config::TILE_SIZE, Config::CHANNEL_TILE_SIZE);
    dim3 blocksPerGrid((Ox2 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (Oy + threadsPerBlock.y - 1) / threadsPerBlock.y,
                       (Nn + threadsPerBlock.z - 1) / threadsPerBlock.z);

    T* d_input;
    T* d_output;
    T (*d_filters)[Ni][Ky][Kx];

    dims input_dims = {NxPad, NyPad, Ni};
    dims output_dims = {Ox, Oy, Nn};


    cudaMalloc((void**)&d_input, I_MEM_SIZE);
    cudaMalloc((void**)&d_output, O_MEM_SIZE);
    cudaMalloc((void**)&d_filters, F_MEM_SIZE);

    // Copy filters and input : host -> device
    gpuErrchk(cudaMemcpy(d_input, h_input, I_MEM_SIZE, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_filters, h_filters, F_MEM_SIZE, cudaMemcpyHostToDevice));
    //gpuErrchk(cudaMemcpy(d_pos_embeds, pos_embeds, PE_MEM_SIZE, cudaMemcpyHostToDevice));


    // Start timer and execute kernel
    cudaStream_t stream;
    cudaStreamCreate(&stream);


    conv_2d_kernel<T, kernel_size><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_input, d_output, d_filters, input_dims, output_dims);
    gpuErrchk(cudaDeviceSynchronize());

    // Copy output : device -> host
    gpuErrchk(cudaMemcpy(h_output, d_output, O_MEM_SIZE, cudaMemcpyDeviceToHost));
}

}

#endif