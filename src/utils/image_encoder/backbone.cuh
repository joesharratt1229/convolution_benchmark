#ifndef BACKBONE_CUH
#define BACKBONE_CUH



namespace image_encoder {

#define TILE_SIZE 32
#define STRIDE 2
#define PAD 1

__constant__ T d_filters[Nn][Ni][Ky][Kx];


template<typename T, int Ni, int Oy, int Ox, int Nn>
__global__ void template_conv_and_bilinear_resid_kernel(T d_backbone_input[Ni][Oy][Ox],  
                                                        T previous_input[Nn][Oy/2][Ox/2],
                                                        T lateral_feature[Nn][Oy][Ox],
                                                        T top_down_feature[Nn][Oy][Ox])
{
    unsigned int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    unsigned int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    unsigned int output_channel = blockIdx.z * blockDim.z + threadIdx.z;

    __shared__ T input_cache[Ni][INPUT_TILE_Y][INPUT_TILE_X];

    if (threadIdx.z < Ni && Stride * threadIdx.y < INPUT_TILE_Y && Stride * threadIdx.x < INPUT_TILE_X) {
        if (threadIdx.y < TILE_SIZE - 1)
            for (int y = 0; y < Stride; y++)
                if (threadIdx.x < TILE_SIZE - 1)
                    for (int x = 0; x < Stride; x++) 
                        input_cache[threadIdx.z][Stride * threadIdx.y + y][Stride * threadIdx.x + x] = d_backbone_input[threadIdx.z][row*Stride + y][col*Stride + x];
                else
                    for (int x = 0; x < Stride + Kx - 1; x++) 
                        input_cache[threadIdx.z][Stride * threadIdx.y + y][Stride * threadIdx.x + x] = d_backbone_input[threadIdx.z][row*Stride + y][col*Stride + x];
        else 
            for (int y = 0; y < Stride + Ky - 1; y++)
                if (threadIdx.x < TILE_SIZE - 1)
                    for (int x = 0; x < Stride; x++) 
                        input_cache[threadIdx.z][Stride * threadIdx.y + y][Stride * threadIdx.x + x] = d_backbone_input[threadIdx.z][row*Stride + y][col*Stride + x];
                else
                    for (int x = 0; x < Stride + Kx - 1; x++) 
                        input_cache[threadIdx.z][Stride * threadIdx.y + y][Stride * threadIdx.x + x] = d_backbone_input[threadIdx.z][row*Stride + y][col*Stride + x];
    }

    __syncthreads();

    T sum1 = 0.0f;

    if (col < Ox && row < Oy && output_channel < Nn) {
        #pragma unroll
        for (int i = 0; i < Ni; i++)
            #pragma unroll
            for (int y = 0; y < Ky; y++)
                #pragma unroll
                for (int x = 0; x < Kx; x++) {
                    T filter_val = d_filters[output_channel][i][y][x];
                    sum1 += input_cache[i][threadIdx.y * Stride + y][(2*threadIdx.x) * Stride + x] * filter_val;
                }
        
        
        lateral_feature[output_channel][row][col] = sum1;
    }

    __syncthreads();

    for (int y = 2*row; y < 2*row + 1; row++)
        for (int x = 2*col; x<2*col+1; col++)
            float origx = x/2;
            float origy = y/2;

            int x0 = static_cast<int>floor(origx);
            int y0 = static_cast<int>floor(origy);
            int x1 = min(x0+1, Ox);
            int y1 = min(y0+1, Oy);

            float dx = origx - x0;
            float dy = origy - y0;

            float value = (previous_input[output_channel][y0][x0] * (1-dx)*(1-dy) + 
                            previous_input[output_channel][y0][x1] * dx*(1-dy) + 
                            previous_input[output_channel][y1][x0] * (1-dx)*dy + 
                            previous_input[output_channel][y1][x1] * dx*xy);

            top_down_feature[output_channel][y][x] = value;

    lateral_feature[output_channel][row][col] += top_down_feature[output_channel][row][col];

}

}

#endif
