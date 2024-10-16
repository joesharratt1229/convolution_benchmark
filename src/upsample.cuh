
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


template <typename T>
__device__ __inline__ T cubic_upsample_1d(T x_coord, T y_coord, T neighbour_coords[4][4]) 
{
   T x_coeffs = get_upsample_coefficients(x_coord);
   T y_coeffs = get_upsample_coefficients(y_coord);

   T result = 0;
   for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
         result += x_coeffs[i] * y_coeffs[j] * neighbour_coords[i][j];
      }
   }
   return result;
}


template <typename T>
__global__ void bicubic_interpolation_kernel(T* input, T* output, int pos_embed_spatial_size) {
    int output_col = blockIdx.x * blockDim.x + threadIdx.x;
    int output_row = blockIdx.y * blockDim.y + threadIdx.y;

    T x_coord = output_col/pos_embed_spatial_size;
    T y_coord = output_row/pos_embed_spatial_size;

    
}