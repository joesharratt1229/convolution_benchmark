#include <cmath>

template<typename T>
__host__
void convolution_cpu(T h_input[Ni][NyPad][NxPad], T h_filters[Nn][Ni][Ky][Kx], T h_output_cpu[Nn][Oy][Ox]) {
    for (int nn = 0; nn < Nn; ++nn) {
        for (int oy = 0; oy < Oy; ++oy) {
            for (int ox = 0; ox < Ox; ++ox) {
                T sum = 0.0f;
                for (int ni = 0; ni < Ni; ++ni) {
                    for (int ky = 0; ky < Ky; ++ky) {
                        for (int kx = 0; kx < Kx; ++kx) {
                            int iy = oy * StrideY + ky;
                            int ix = ox * StrideX + kx;
                            sum += h_input[ni][iy][ix] * h_filters[nn][ni][ky][kx];
                        }
                    }
                }
                h_output_cpu[nn][oy][ox] = sum;
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


template<typename T>
__inline__ T cubicKernel(T x) {
    T A = -0.75;

    if (x <= 1) return ((A + 2) * x - (A + 3)) * x * x + 1;
    else if (x < 2) return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
    return 0;
}

template<typename T>
__inline__ void get_upsample_coefficients_cpu(T x1, T* coeffs) {
    coeffs[0] = cubicKernel<T>(x1 + 1);
    coeffs[1] = cubicKernel<T>(x1);

    T x2 = 1 - x1;
    coeffs[2] = cubicKernel<T>(x2);
    coeffs[3] = cubicKernel<T>(x2 + 1);
} 


template<typename T>
__host__
void bicubic_convolution_cpu(T pos_embeds[POS_EMBEDS][POS_EMBEDS], const int height, const int width, T h_output_cpu[Nn][Oy][Ox]) 
{
    T scale_factor_x = static_cast<T>(width) / static_cast<T>(POS_EMBEDS);
    T scale_factor_y = static_cast<T>(height) / static_cast<T>(POS_EMBEDS);

    for (int nn = 0; nn < Nn; nn++) {
        for (int oy = 0; oy < Oy; oy++) {
            for (int ox = 0; ox < Ox; ox++) {
                T sum = 0.0f;
                T iy_pos = oy / scale_factor_y;
                T ix_pos = ox / scale_factor_x;

                int iy_floor = std::floor(iy_pos);
                int ix_floor = std::floor(ix_pos);

                T scaled_x_coord = ix_pos - ix_floor;
                T scaled_y_coord = iy_pos - iy_floor;

                T x_coeffs[4];
                T y_coeffs[4];
                
                get_upsample_coefficients_cpu(scaled_x_coord, x_coeffs);
                get_upsample_coefficients_cpu(scaled_y_coord, y_coeffs);

                for (int y = 0; y < 4; y++) {
                    for (int x = 0; x < 4; x++) {
                        int py = std::min(std::max(iy_floor + y - 1, 0), POS_EMBEDS - 1);
                        int px = std::min(std::max(ix_floor + x - 1, 0), POS_EMBEDS - 1);
                        sum += pos_embeds[py][px] * x_coeffs[x] * y_coeffs[y];
                    }
                }

                h_output_cpu[nn][oy][ox] = sum;
            }
        }
    }
}