#include "common.h"

template<typename T>
__device__ __inline__ T cubic_convolution_1(T x, T a) {
    return (a+2)*std::pow(x,3) - (a+3)*std::pow(x,2) + 1;
}

template<typename T>
__device__ __inline__ T cubic_convolution_2(T x, T a) {
    return a*std::pow(x,3) - 2*a*std::pow(x,2) + x;
}

template<typename T>
__device__ __inline__ T get_upsample_coefficients(T x1) {
    T A = -0.75;
    T coeffs[4];
    coeffs[0] = cubic_convolution_2<T>(x1+1, A);
    coeffs[1] = cubic_convolution_1<T>(x1, A);

    T x2 = 1 - x1;
    coeffs[2] = cubic_convolution_1<T>(x2, A);
    coeffs[3] = cubic_convolution_2<T>(x2 + 1, A);
} 

template<typename T>
__device__ __inline__ void get_upsample_coefficients(T x1) {
    T A = -0.75;
    T coeffs[4];
    coeffs[0] = cubic_convolution_2<T>(x1+1, A);
    coeffs[1] = cubic_convolution_1<T>(x1, A);

    T x2 = 1 - x1;
    coeffs[2] = cubic_convolution_1<T>(x2, A);
    coeffs[3] = cubic_convolution_2<T>(x2 + 1, A);
} 


template<typename T>
__device__ __inline__ T upsample_value_bounded(T data,
                                                    int width,
                                                    int height,
                                                    int channel,
                                                    int y,
                                                    int x)
{
    int access_y = max(min(y, height - 1), 0);
    int access_x = max(min(x, width - 1), 0);
    return data[channel][access_y][access_x];
}



template <typename T>
__global__ void bicubic_interpolation_kernel(T* input, 
                                             T* output, 
                                             dims input_dims,
                                             dims output_dims,
                                             const int scale_factor) {
    int output_col = blockIdx.x * blockDim.x + threadIdx.x;
    int output_row = blockIdx.y * blockDim.y + threadIdx.y;
    int output_channel = blockIdx.z * blockDim.z + threadIdx.z;

    T x_coord = output_col/scale_factor;
    T y_coord = output_row/scale_factor;

    if (output_dims.width == input_dims.width && output_dims.height == input_dims.height) {
        output[output_channel][output_row][output_col] = input[output_channel][output_row][output_col];
        return;
    }

    T x_floor = floor(x_coord);
    T y_floor = floor(y_coord);

    T scaled_x_coord = x_coord - x_floor;
    T scaled_y_coord = y_coord - y_floor;


    T x_coeffs[4];
    T y_coeffs[4];

    get_upsample_coefficients(scaled_x_coord, x_coeffs);
    get_upsample_coefficients(scaled_y_coord, y_coeffs);

    T output_value = 0;
    for (int j = 0; j < 4; j++) {
        for (int i = 0; i < 4; i++) {
            T value = upsample_value_bounded(input, 
                                             input_dims.width,
                                             input_dims.height,
                                             output_channel,
                                             x_floor - 1 + i,
                                             y_floor - 1 + j);
            output_value += x_coeffs[i] * y_coeffs[j] * value;
        }
    }
    output[output_channel][output_row][output_col] = output_value;  


}