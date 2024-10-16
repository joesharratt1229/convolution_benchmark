#include <string>
#include <cmath>

#include "common.h" 
#include "convolution.cuh"

using namespace std;

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


template<typename T>
__host__ void randomizeFilters(T h_filters[Nn][Ni][Ky][Kx]);
template<typename T>
__host__ void randomizeInput(T h_input[Ni][NyPad][NxPad]);
template<typename T>
__host__ void padInput(T h_input[Ni][NyPad][NxPad]);
template<typename T>
__host__ void printParameters();
template<typename T>
__host__ void convolution_cpu(T h_input[Ni][NyPad][NxPad], T h_filters[Nn][Ni][Ky][Kx], T h_output_cpu[Nn][Oy][Ox]);
template<typename T>
__host__ void checkOutput(T *h_output, T *h_output_cpu, unsigned int total_size);


int main(int argc, char **argv) {
    bool DEBUG = ((argc > 1) && (std::string(argv[1]) == "--debug"));

    unsigned int Ox2 = (Ox + 1) / 2;

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE, CHANNEL_SIZE);
    dim3 blocksPerGrid((Ox2 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (Oy + threadsPerBlock.y - 1) / threadsPerBlock.y,
                       (Nn + threadsPerBlock.z - 1) / threadsPerBlock.z);


    static floatT h_input[Ni][NyPad][NxPad];
    static floatT h_output[Nn][Oy][Ox];
    static floatT h_output_cpu[Nn][Oy][Ox];
    static floatT h_filters[Nn][Ni][Ky][Kx]; 

    floatT (*d_input)[NyPad][NxPad];
    floatT (*d_output)[Oy][Ox];
    floatT (*d_filters)[Ni][Ky][Kx];


    cudaMalloc((void**)&d_input, I_MEM_SIZE);
    cudaMalloc((void**)&d_output, O_MEM_SIZE);
    cudaMalloc((void**)&d_filters, F_MEM_SIZE);


    // Randomize inputs/filters and set padded regions to 0
    randomizeFilters(h_filters);
    randomizeInput(h_input);
    padInput(h_input);

    // Copy filters and input : host -> device
    gpuErrchk(cudaMemcpy(d_input, h_input, I_MEM_SIZE, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_filters, h_filters, F_MEM_SIZE, cudaMemcpyHostToDevice));


    // Start timer and execute kernel
    cudaStream_t stream;
    cudaStreamCreate(&stream);


    conv_2d<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_filters, d_output);

    gpuErrchk(cudaDeviceSynchronize());

    // Copy output : device -> host
    gpuErrchk(cudaMemcpy(h_output, d_output, O_MEM_SIZE, cudaMemcpyDeviceToHost));

    // Check output
    if (DEBUG) {
        convolution_cpu(h_input, h_filters, h_output_cpu);
        checkOutput(&h_output[0][0][0], &h_output_cpu[0][0][0], Ox * Oy * Nn);
    } 

    return 0;
} 


template<typename T>
__host__
void randomizeFilters(T h_filters[Nn][Ni][Ky][Kx]) {
    for (int yy = 0; yy < Ky; ++yy)
        for (int xx = 0; xx < Kx; ++xx)
            for (int nn = 0; nn < Nn; ++nn)
                for (int ni = 0; ni < Ni; ++ni)
                    h_filters[nn][ni][yy][xx] = static_cast<T>(static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f);
}

template<typename T>
__host__
void randomizeInput(T h_input[Ni][NyPad][NxPad]) {
    for (int ni = 0; ni < Ni; ++ni)
        for (int yy = 0; yy < NyPad; ++yy)
            for (int xx = 0; xx < NxPad; ++xx)
                h_input[ni][yy][xx] = static_cast<T>(static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f);
}

template<typename T>
__host__
void padInput(T h_input[Ni][NyPad][NxPad]) {
    // Set padded regions to 0
    for (int z = 0; z < Ni; z++) {
            for (int x = 0; x < NxPad; x++) {
                h_input[z][0][x] = 0;
                h_input[z][NyPad - 1][x] = 0;
            }
            for (int y = 0; y < NyPad; y++) {
                h_input[z][y][0] = 0;
                h_input[z][y][NxPad - 1] = 0;
            }
    }
}


template<typename T>
__host__
void printParameters() {
    printf("\n\n");
    printf("Padding: %d\n", Pad);
    printf("Stride (StrideX, StrideY): (%d, %d)\n", StrideX, StrideY);

    printf("\n\n");
    printf("Input dimensions (Nx, Ny, Ni): (%d, %d, %d)\n", Nx, Ny, Ni);
    printf("Input dimensions with Pad (Nx+%d, Ny+%d, Ni): (%d, %d, %d)\n", (2 * Pad), (2 * Pad), NxPad, NyPad,
           Ni);
    printf("Input number of elements: %dx%dx%d = %d\n", Nx, Ny, Ni, Nx * Ny * Ni);
    printf("Input memory size: %lu bytes\n", I_MEM_SIZE);

    printf("\n\n");
    printf("Output dimensions (Ox, Oy, Nn): (%d, %d, %d)\n", Ox, Oy, Nn);
    printf("Output number of elements: %dx%dx%d = %d\n", Ox, Oy, Nn, Ox * Oy * Nn);
    printf("Output memory size: %lu bytes\n", O_MEM_SIZE);

    printf("\n\n");
    printf("Weights dimensions (Kx, Ky, Ni, Nn): (%d, %d, %d, %d)\n", Kx, Ky, Ni, Nn);
    printf("Weights number of elements: %dx%dx%dx%d = %d\n", Kx, Ky, Ni, Nn, Kx * Ky * Ni * Nn);
    printf("Weights memory size: %lu bytes\n", F_MEM_SIZE);
}



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