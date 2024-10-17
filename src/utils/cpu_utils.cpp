


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