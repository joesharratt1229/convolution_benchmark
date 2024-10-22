#include "cuda_runtime.h"

#include "common.h"
#include "gpu_utils.cuh"

template<typename T>
__global__ void conv_2d_kernel(T d_input[Ni][NyPad][NxPad], 
                               T d_filters[Nn][Ni][Ky][Kx], 
                               T d_output[Nn][Oy][Ox])
{
    unsigned int col = 2*(blockIdx.x * TILE_SIZE + threadIdx.x);
    unsigned int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    unsigned int output_channel = blockIdx.z * blockDim.z + threadIdx.z;

    __shared__ T input_cache[Ni][INPUT_TILE_Y][INPUT_TILE_X*2];

    if (threadIdx.z < Ni && StrideY * threadIdx.y < INPUT_TILE_Y && StrideX * threadIdx.x < INPUT_TILE_X) {
        if (threadIdx.y < TILE_SIZE - 1)
            for (int y = 0; y < StrideY; y++)
                if (threadIdx.x < TILE_SIZE - 1)
                    for (int x = 0; x < 2* StrideX; x++) 
                        input_cache[threadIdx.z][StrideY * threadIdx.y + y][2*StrideX * threadIdx.x + x] = d_input[threadIdx.z][row*StrideY + y][col*StrideX + x];
                else
                    for (int x = 0; x < 2* StrideX + Kx - 1; x++) 
                        input_cache[threadIdx.z][StrideY * threadIdx.y + y][2*StrideX * threadIdx.x + x] = d_input[threadIdx.z][row*StrideY + y][col*StrideX + x];
        else 
            for (int y = 0; y < StrideY + Ky - 1; y++)
                if (threadIdx.x < TILE_SIZE - 1)
                    for (int x = 0; x < 2* StrideX; x++) 
                        input_cache[threadIdx.z][StrideY * threadIdx.y + y][2*StrideX * threadIdx.x + x] = d_input[threadIdx.z][row*StrideY + y][col*StrideX + x];
                else
                    for (int x = 0; x < 2* StrideX + Kx - 1; x++) 
                        input_cache[threadIdx.z][StrideY * threadIdx.y + y][2*StrideX * threadIdx.x + x] = d_input[threadIdx.z][row*StrideY + y][col*StrideX + x];
    }
    
    __syncthreads();

    T sum1 = 0.0f;
    T sum2 = 0.0f;

    if (row < Oy && output_channel < Nn) {
        #pragma unroll
        for (int i = 0; i < Ni; i++)
            #pragma unroll
            for (int y = 0; y < Ky; y++)
                #pragma unroll
                for (int x = 0; x < Kx; x++) {
                    T filter_val = d_filters[output_channel][i][y][x];
                    sum1 += input_cache[i][threadIdx.y * StrideY + y][(2*threadIdx.x) * StrideX + x] * filter_val;
                    sum2 += input_cache[i][threadIdx.y * StrideY + y][(2*threadIdx.x+1) * StrideX + x] * filter_val;

                }
        
        if (col < Ox) {
            d_output[output_channel][row][col] = sum1;
        }
        if (col+1 < Ox) {
            d_output[output_channel][row][col+1] = sum2;
        }
    }
}



template<typename T, int CHANNEL_SIZE>
__host__ void template_conv_2d(T h_input[Ni][NyPad][NxPad], 
                               T h_filters[Nn][Ni][Ky][Kx], 
                               T h_output[Nn][Oy][Ox])
{
    unsigned int Ox2 = (Ox + 1) / 2;

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE, CHANNEL_SIZE);
    dim3 blocksPerGrid((Ox2 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (Oy + threadsPerBlock.y - 1) / threadsPerBlock.y,
                       (Nn + threadsPerBlock.z - 1) / threadsPerBlock.z);

    T (*d_input)[NyPad][NxPad];
    T (*d_output)[Oy][Ox];
    T (*d_filters)[Ni][Ky][Kx];


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


    conv_2d_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_input, d_filters, d_output);
    gpuErrchk(cudaDeviceSynchronize());

    // Copy output : device -> host
    gpuErrchk(cudaMemcpy(h_output, d_output, O_MEM_SIZE, cudaMemcpyDeviceToHost));
}