#include <cmath>
#include <vector>
#include <algorithm>

template<typename T, int Nz_input, int Ny_input, int Nx_input, int Nz_output, int Ny_output, int Nx_output, int kernel_size>
__host__
void convolution_cpu(T h_input[Nz_input][Ny_input][Nx_input], 
                     T h_filters[Nz_output][Nz_input][kernel_size][kernel_size], 
                     T h_output_cpu[Nz_output][Ny_output][Nx_output]) {
    using Config = TileConfig<kernel_size>;

    for (int nn = 0; nn < Nz_output; ++nn) {
        for (int oy = 0; oy < Ny_output; ++oy) {
            for (int ox = 0; ox < Nx_output; ++ox) {
                T sum = 0.0f;
                for (int ni = 0; ni < Nz_input; ++ni) {
                    for (int ky = 0; ky < kernel_size; ++ky) {
                        for (int kx = 0; kx < kernel_size; ++kx) {
                            int iy = oy * Config::STRIDE + ky;
                            int ix = ox * Config::STRIDE + kx;
                            sum += h_input[ni][iy][ix] * h_filters[nn][ni][ky][kx];
                        }
                    }
                }
                h_output_cpu[nn][oy][ox] = sum;
            }
        }
    }
}


template<typename T, int N_channels, int N_height, int N_width>
void bilinear_interpolation_2x(T h_input[N_channels][N_height/2][N_width/2], 
                               T h_backbone_output[N_channels][N_height][N_width], 
                               T h_1x1_output[N_channels][N_height][N_width]) {
    for (int nn = 0; nn < N_channels; nn++) {
        for (int y = 0; y < N_height; y++) {
            for (int x = 0; x < N_width; x++) {

                float origx = static_cast<float>(x)/2;
                float origy = static_cast<float>(y)/2;

                int x0 = static_cast<int>(floor(origx));
                int y0 = static_cast<int>(floor(origy));

                int x1 = min(x0+1, (N_width/2)-1);
                int y1 = min(y0+1, (N_height/2)-1);

                float dx = origx - x0;
                float dy = origy - y0;

                h_backbone_output[nn][y][x] = (h_input[nn][y0][x0] * (1-dx)*(1-dy) + 
                                     h_input[nn][y0][x1] * dx*(1-dy) + 
                                     h_input[nn][y1][x0] * (1-dx)*dy + 
                                     h_input[nn][y1][x1] * dx*dy);
                
                h_1x1_output[nn][y][x] += h_backbone_output[nn][y][x];
            }
        }
    }
}



template<typename T>
__host__
void checkOutput(T *h_output, T *h_output_cpu, unsigned int total_size) {
    for (int i = 0; i < total_size; i++) {
        float gpu_val = static_cast<float>(h_output[i]);
        float cpu_val = static_cast<float>(h_output_cpu[i]);
        if (std::abs(gpu_val - cpu_val) > 1e-3) {
            printf("Mismatch at h_output[%d]: %f (CPU) vs %f (GPU)\n", i, cpu_val, gpu_val);
            exit(1);
        }
    }
}

__inline__ float cubicKernel(float x) {
    float A = -0.75;

    if (x <= 1) return ((A + 2) * x - (A + 3)) * x * x + 1;
    else if (x < 2) return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
    return 0;
}

__inline__ void get_upsample_coefficients_cpu(float x1, float* coeffs) {
    coeffs[0] = cubicKernel(x1 + 1);
    coeffs[1] = cubicKernel(x1);

    float x2 = 1 - x1;
    coeffs[2] = cubicKernel(x2);
    coeffs[3] = cubicKernel(x2 + 1);
} 


template<typename T>
void bicubic_convolution_cpu(T pos_embeds[Nn][POS_EMBEDS][POS_EMBEDS], const int height, const int width, T h_output_cpu[Nn][Oy][Ox]) 
{
    float scale_factor_x =  width / POS_EMBEDS;
    float scale_factor_y = height / POS_EMBEDS;

    for (int nn = 0; nn < Nn; nn++) {
        for (int oy = 0; oy < Oy; oy++) {
            for (int ox = 0; ox < Ox; ox++) {
                float sum = 0.0f;
                float iy_pos = oy / scale_factor_y;
                float ix_pos = ox / scale_factor_x;

                int iy_floor = std::floor(iy_pos);
                int ix_floor = std::floor(ix_pos);

                float scaled_x_coord = ix_pos - ix_floor;
                float scaled_y_coord = iy_pos - iy_floor;

                float x_coeffs[4];
                float y_coeffs[4];
                
                get_upsample_coefficients_cpu(scaled_x_coord, x_coeffs);
                get_upsample_coefficients_cpu(scaled_y_coord, y_coeffs);

                for (int y = 0; y < 4; y++) {
                    for (int x = 0; x < 4; x++) {
                        int py = std::min(std::max(iy_floor + y - 1, 0), POS_EMBEDS - 1);
                        int px = std::min(std::max(ix_floor + x - 1, 0), POS_EMBEDS - 1);
                        sum += static_cast<float>(pos_embeds[nn][py][px]) * x_coeffs[x] * y_coeffs[y];
                    }
                }

                h_output_cpu[nn][oy][ox] = static_cast<T>(sum);
            }
        }
    }
}
