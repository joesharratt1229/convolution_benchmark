#ifndef UPSAMPLE_CUH
#define UPSAMPLE_CUH


#include "common.h"
#include "gpu_utils.cuh"

template<typename accFloatT>
__device__ __inline__ accFloatT cubic_convolution_1(accFloatT x, accFloatT A) {
    return ((A + 2) * x - (A + 3)) * x * x + 1;
}

template<typename accFloatT>
__device__ __inline__ accFloatT cubic_convolution_2(accFloatT x, accFloatT A) {
    return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
}

template<typename accFloatT>
__device__ __inline__ void get_upsample_coefficients(accFloatT x1, accFloatT* coeffs) {
    accFloatT A = -0.75;
    coeffs[0] = cubic_convolution_2<accFloatT>(x1+1, A);
    coeffs[1] = cubic_convolution_1<accFloatT>(x1, A);

    accFloatT x2 = 1 - x1;
    coeffs[2] = cubic_convolution_1<accFloatT>(x2, A);
    coeffs[3] = cubic_convolution_2<accFloatT>(x2 + 1, A);
} 


template<typename T>
__device__ __inline__ T upsample_value_bounded(T* data,
                                              int width,
                                              int height,
                                              int channel,
                                              int x,
                                              int y)
{
    int access_y = max(min(y, height - 1), 0);
    int access_x = max(min(x, width - 1), 0);
    return data[channel * width * height + access_y * width + access_x];
}


template <typename T>
__global__ void bicubic_interpolation_kernel(T* input, 
                                             T* output, 
                                             dims input_dims,
                                             dims output_dims,
                                             const accFloatT scale_factor_x,
                                             const accFloatT scale_factor_y) {
                                   
    int output_col = blockIdx.x * blockDim.x + threadIdx.x;
    int output_row = blockIdx.y * blockDim.y + threadIdx.y;
    int output_channel = blockIdx.z * blockDim.z + threadIdx.z;

    if (output_col >= output_dims.width || output_row >= output_dims.height || output_channel >= output_dims.channel) {
        return;
    }

    accFloatT x_coord = output_col/scale_factor_x;
    accFloatT y_coord = output_row/scale_factor_y;

    if (output_dims.width == input_dims.width && output_dims.height == input_dims.height) {
        output[output_channel * output_dims.width * output_dims.height + output_row * output_dims.width + output_col] = input[output_channel * input_dims.width * input_dims.height + output_row * input_dims.width + output_col];
        return;
    }

    int x_floor = floor(x_coord);
    int y_floor = floor(y_coord);

    accFloatT scaled_x_coord = x_coord - x_floor;
    accFloatT scaled_y_coord = y_coord - y_floor;


    accFloatT x_coeffs[4];
    accFloatT y_coeffs[4];

    get_upsample_coefficients<accFloatT>(scaled_x_coord, x_coeffs);
    get_upsample_coefficients<accFloatT>(scaled_y_coord, y_coeffs);

    accFloatT output_value = 0;

    for (int j = 0; j < 4; j++) {
        for (int i = 0; i < 4; i++) {
            T value = upsample_value_bounded(input, 
                                             input_dims.width,
                                             input_dims.height,
                                             output_channel,
                                             x_floor - 1 + i,
                                             y_floor - 1 + j);
            output_value += x_coeffs[i] * y_coeffs[j] * static_cast<accFloatT>(value);
        }
    }
    output[output_channel * output_dims.width * output_dims.height + output_row * output_dims.width + output_col] = output_value;  


}


template <typename T, int PosEmbeds, int OutNn, int OutOy, int OutOx, int CHANNEL_SIZE>
__host__ void template_bicubic_upsample_and_window_embed(T input[OutNn][PosEmbeds][PosEmbeds], 
                                                         T output[OutNn][OutOy][OutOx], 
                                                        dims input_dims,
                                                        dims output_dims) {                                
    

    accFloatT scale_factor_x = output_dims.width / input_dims.width;
    accFloatT scale_factor_y = output_dims.height / input_dims.height;

    T *d_input[nStreams];
    T *d_output[nStreams];

    cudaStream_t stream[nStreams];

    int streamChannelSize = (output_dims.channel + nStreams - 1) / nStreams;

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE, CHANNEL_SIZE);
    dim3 blocksPerGrid((output_dims.width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                        (output_dims.height + threadsPerBlock.y - 1) / threadsPerBlock.y,
                        (streamChannelSize + threadsPerBlock.z - 1) / threadsPerBlock.z);


    for (int i = 0; i < nStreams; i++) {
        cudaStreamCreate(&stream[i]);

        cudaMalloc((void**)&d_input[i], sizeof(T) * PosEmbeds * PosEmbeds * streamChannelSize);
        cudaMalloc((void**)&d_output[i], sizeof(T) * OutOy * OutOx * streamChannelSize);

        gpuErrchk(cudaMemcpyAsync(d_input[i], &input[i*streamChannelSize][0][0], sizeof(T) * PosEmbeds * PosEmbeds * streamChannelSize, cudaMemcpyHostToDevice, stream[i]));
        bicubic_interpolation_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream[i]>>>(d_input[i], d_output[i], input_dims, output_dims, scale_factor_x, scale_factor_y);
        gpuErrchk(cudaMemcpyAsync(&output[i*streamChannelSize][0][0], d_output[i], sizeof(T) * OutOy * OutOx * streamChannelSize, cudaMemcpyDeviceToHost, stream[i]));

        cudaStreamSynchronize(stream[i]);
        cudaStreamDestroy(stream[i]);
        cudaFree(d_input[i]);
        cudaFree(d_output[i]);
    }

}

#endif