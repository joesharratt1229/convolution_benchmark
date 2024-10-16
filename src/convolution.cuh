#include "common.h"


template<typename T>
__global__ void conv_2d(T d_input[Ni][NyPad][NxPad], T d_filters[Nn][Ni][Ky][Kx], T d_output[Nn][Oy][Ox])
{
        unsigned int col = 2*(blockIdx.x * TILE_SIZE + threadIdx.x);
    unsigned int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    unsigned int output_channel = blockIdx.z * CHANNEL_SIZE + threadIdx.z;

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