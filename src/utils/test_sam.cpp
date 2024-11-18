#include <cmath>
#include <vector>
#include <algorithm>

#include "utils/common.h"


template <typename T, typename accT>
void multiHeadAttention_cpu(T* query, T* key, T* value, T* output,
                           int seq_len, int embed_dim, int num_heads) {
    const accT scale = static_cast<accT>(1.0f) / sqrt(static_cast<accT>(embed_dim));
    
    accT* scores = new accT[num_heads * seq_len * seq_len];
    accT* temp_output = new accT[num_heads * seq_len * embed_dim];
    
    for (int head_id = 0; head_id < num_heads; head_id++) {
        for (int seq_id = 0; seq_id < seq_len; seq_id++) {
            for (int j = 0; j < seq_len; j++) {
                accT sum = 0;
                for (int d = 0; d < embed_dim; d++) {
                    int q_idx = head_id * seq_len * embed_dim + seq_id * embed_dim + d;
                    int k_idx = head_id * seq_len * embed_dim + j * embed_dim + d;
                    sum += static_cast<accT>(query[q_idx]) * static_cast<accT>(key[k_idx]);
                }
                scores[head_id * seq_len * seq_len + seq_id * seq_len + j] = sum * scale;
            }
        }
        
        for (int seq_id = 0; seq_id < seq_len; seq_id++) {
            accT warp_max = scores[head_id * seq_len * seq_len + seq_id * seq_len];
            for (int j = 1; j < seq_len; j++) {
                int score_idx = head_id * seq_len * seq_len + seq_id * seq_len + j;
                warp_max = std::max(warp_max, scores[score_idx]);
            }
            
            accT* temp_exp = new accT[seq_len];
            for (int j = 0; j < seq_len; j++) {
                int score_idx = head_id * seq_len * seq_len + seq_id * seq_len + j;
                temp_exp[j] = expf(scores[score_idx] - warp_max);
            }
            
            accT sum_exp = 0;
            for (int j = 0; j < seq_len; j++) {
                sum_exp += temp_exp[j];
            }
            
            for (int j = 0; j < seq_len; j++) {
                int score_idx = head_id * seq_len * seq_len + seq_id * seq_len + j;
                scores[score_idx] = temp_exp[j] / (sum_exp + static_cast<accT>(1e-6));
            }
            
            delete[] temp_exp;
        }
        
        for (int seq_id = 0; seq_id < seq_len; seq_id++) {
            for (int d = 0; d < embed_dim; d++) {
                accT sum = 0;
                for (int j = 0; j < seq_len; j++) {
                    int score_idx = head_id * seq_len * seq_len + seq_id * seq_len + j;
                    int v_idx = head_id * seq_len * embed_dim + j * embed_dim + d;
                    sum += scores[score_idx] * static_cast<accT>(value[v_idx]);
                }
                int out_idx = head_id * seq_len * embed_dim + seq_id * embed_dim + d;
                temp_output[out_idx] = sum;
            }
        }
    }
    
    for (int i = 0; i < num_heads * seq_len * embed_dim; i++) {
        output[i] = static_cast<T>(temp_output[i]);
    }
    
    delete[] scores;
    delete[] temp_output;
}

template<typename T, int kernel_size>
void convolution_cpu(T* h_input, 
                     T* h_filters, 
                     T* h_bias,
                     T* h_output_cpu,
                     Dimensions input_dims,
                     Dimensions output_dims) {
    using Config = TileConfig<kernel_size>;

    for (int nn = 0; nn < output_dims.num_channels; ++nn) {
        for (int oy = 0; oy < output_dims.y_dimension; ++oy) {
            for (int ox = 0; ox < output_dims.x_dimension; ++ox) {
                T sum = 0.0f;
                for (int ni = 0; ni < input_dims.num_channels; ++ni) {
                    for (int ky = 0; ky < kernel_size; ++ky) {
                        for (int kx = 0; kx < kernel_size; ++kx) {
                            int iy = oy * Config::STRIDE + ky;
                            int ix = ox * Config::STRIDE + kx;
                            sum += h_input[ni * input_dims.y_dimension * input_dims.x_dimension + iy * input_dims.x_dimension + ix] * h_filters[nn * input_dims.num_channels * kernel_size * kernel_size + ni * kernel_size * kernel_size + ky * kernel_size + kx];
                        }
                    }
                }
                h_output_cpu[nn * output_dims.y_dimension * output_dims.x_dimension + oy * output_dims.x_dimension + ox] = sum + h_bias[nn];
            }
        }
    }
}


template<typename T>
void pos_embed_cpu(T* h_output_cpu,
                   Dimensions output_dims) {

    for (int nn = 0; nn < output_dims.num_channels; ++nn) {
        for (int oy = 0; oy < output_dims.y_dimension; ++oy) {
            for (int ox = 0; ox < output_dims.x_dimension; ++ox) {
                accFloatT y_embed = oy+1;
                accFloatT x_embed = ox+1;

                y_embed = y_embed/(output_dims.y_dimension + EPSILON) * SCALE;
                x_embed = x_embed/(output_dims.x_dimension + EPSILON) * SCALE;

                accFloatT power_term = 2*(floorf((nn)/2))/output_dims.num_channels;
                accFloatT d_dimensions_x = std::pow(TEMPERATURE, power_term);
                accFloatT d_dimensions_y = std::pow(TEMPERATURE, power_term);

                const bool is_even = nn%2 == 0;
                const bool is_first_half = nn < output_dims.num_channels/2;
                const accFloatT embed_val = is_first_half ? y_embed : x_embed;
                const accFloatT dim = is_first_half ? d_dimensions_y : d_dimensions_x;

                accFloatT val = is_even ? std::sin(embed_val / dim) : std::cos(embed_val / dim);
                h_output_cpu[nn * output_dims.y_dimension * output_dims.x_dimension + oy * output_dims.x_dimension + ox] = static_cast<T>(val);
            }
        }
    }
}





template<typename T>
void bilinear_interpolation_2x(T* h_input,  
                               T* h_1x1_output,
                               Dimensions input_dims,
                               Dimensions output_dims) {

    T* h_backbone_output = new T[output_dims.num_channels * output_dims.y_dimension * output_dims.x_dimension];
    assert(input_dims.x_dimension * 2 == output_dims.x_dimension);
    assert(input_dims.y_dimension * 2 == output_dims.y_dimension);

    for (int nn = 0; nn < output_dims.num_channels; nn++) {
        for (int y = 0; y < output_dims.y_dimension; y++) {
            for (int x = 0; x < output_dims.x_dimension; x++) {

                accFloatT origx = static_cast<accFloatT>(x)/2;
                accFloatT origy = static_cast<accFloatT>(y)/2;

                int x0 = static_cast<int>(floor(origx));
                int y0 = static_cast<int>(floor(origy));

                int x1 = std::min(x0+1, (input_dims.x_dimension)-1);
                int y1 = std::min(y0+1, (input_dims.y_dimension)-1);

                accFloatT dx = origx - x0;
                accFloatT dy = origy - y0;

                h_backbone_output[nn * output_dims.y_dimension * output_dims.x_dimension + y * output_dims.x_dimension + x] = ((1-dx)*(1-dy) * static_cast<accFloatT>(h_input[nn * input_dims.y_dimension * input_dims.x_dimension + y0 * input_dims.x_dimension + x0]) + 
                                               dx*(1-dy) * static_cast<accFloatT>(h_input[nn * input_dims.y_dimension * input_dims.x_dimension + y0 * input_dims.x_dimension + x1]) + 
                                               (1-dx)*dy * static_cast<accFloatT>(h_input[nn * input_dims.y_dimension * input_dims.x_dimension + y1 * input_dims.x_dimension + x0]) + 
                                               dx*dy * static_cast<accFloatT>(h_input[nn * input_dims.y_dimension * input_dims.x_dimension + y1 * input_dims.x_dimension + x1]));
                
                h_1x1_output[nn * output_dims.y_dimension * output_dims.x_dimension + y * output_dims.x_dimension + x] += h_backbone_output[nn * output_dims.y_dimension * output_dims.x_dimension + y * output_dims.x_dimension + x];
            }
        }
    }

    delete[] h_backbone_output;
}



template<typename T>
__host__
void checkOutput(T *h_output, T *h_output_cpu, unsigned int total_size) {
    for (int i = 0; i < total_size; i++) {
        float gpu_val = static_cast<float>(h_output[i]);
        float cpu_val = static_cast<float>(h_output_cpu[i]);
        if (std::abs(gpu_val - cpu_val) > 1e-1) {
            printf("Mismatch at h_output[%d]: %f (CPU) vs %f (GPU)\n", i, cpu_val, gpu_val);
            exit(1);
        }

        if (std::abs(gpu_val - cpu_val) > 1e-2) {
            printf("Mismatch at h_output[%d]: %f (CPU) vs %f (GPU)\n", i, cpu_val, gpu_val);
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


template<typename T, int Num_channels, int x_dim, int y_dim, int output_y_dim, int output_x_dim>
void bicubic_convolution_cpu(T pos_embeds[Num_channels][x_dim][y_dim], 
                             const int height, 
                             const int width, 
                             T h_output_cpu[Num_channels][output_y_dim][output_x_dim]) 
{
    float scale_factor_x =  width / x_dim;
    float scale_factor_y = height / y_dim;

    for (int nn = 0; nn < Num_channels; nn++) {
        for (int oy = 0; oy < output_y_dim; oy++) {
            for (int ox = 0; ox < output_x_dim; ox++) {
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
                        int py = std::min(std::max(iy_floor + y - 1, 0), y_dim - 1);
                        int px = std::min(std::max(ix_floor + x - 1, 0), x_dim - 1);
                        sum += static_cast<float>(pos_embeds[nn][py][px]) * x_coeffs[x] * y_coeffs[y];
                    }
                }

                h_output_cpu[nn][oy][ox] = static_cast<T>(sum);
            }
        }
    }
}
