/*
V100_CUDA_CAP 7.0
V100_GLOBAL_MEM_TOTAL 12621381632
V100_SM_COUNT 80
V100_CUDA_CORES_PER_SM 64
V100_CUDA_CORES_TOTAL 5120
V100_L2_SIZE 4718592
V100_SH_MEM_PER_BLOCK 49152
V100_REGS_PER_BLOCK 65536
V100_WARP_SIZE 32
V100_MAX_THREADS_PER_SM 2048
V100_MAX_THREADS_PER_BLOCK 1024
 */

#include <string>
#include <cmath>

#define Pad 1
#define StrideX 1
#define StrideY 1
#define NxPad (Nx + (2*Pad))
#define NyPad (Ny + (2*Pad))
#define Ox (((Nx - Kx + 2*Pad) / StrideX) + 1)
#define Oy (((Ny - Ky + 2*Pad) / StrideY) + 1)
#define I_SIZE (Ni * NyPad * NxPad)
#define O_SIZE (Nn * Oy * Ox)
#define F_SIZE (Nn * Ni * Ky * Kx)
#define I_MEM_SIZE (I_SIZE * sizeof(float))
#define O_MEM_SIZE (O_SIZE * sizeof(float))
#define F_MEM_SIZE (F_SIZE * sizeof(float))
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

using namespace std;

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void conv_2d();

__host__ void randomizeFilters();
__host__ void randomizeInput();
__host__ void padInput();
__host__ void printParameters();
__host__ void convolution_cpu();
__host__ void checkOutput();

__device__ float d_input[Ni][NyPad][NxPad];
__device__ float d_output[Nn][Oy][Ox];
__device__ float d_filters[Nn][Ni][Ky][Kx];

float h_input[Ni][NyPad][NxPad];
float h_output[Nn][Oy][Ox];
float h_filters[Nn][Ni][Ky][Kx];
float h_output_cpu[Nn][Oy][Ox];

int main(int argc, char **argv) {
    bool DEBUG = ((argc > 1) && (std::string(argv[1]) == "--debug"));

    dim3 blocksPerGrid(Ox, Oy, 1);
    dim3 threadsPerBlock(1, 1, Nn);

    // Randomize inputs/filters and set padded regions to 0
    randomizeFilters();
    randomizeInput();
    padInput();

    // Copy filters and input : host -> device
    gpuErrchk(cudaMemcpyToSymbol(d_input, h_input, I_MEM_SIZE));
    gpuErrchk(cudaMemcpyToSymbol(d_filters, h_filters, F_MEM_SIZE));


    // Start timer and execute kernel
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    conv_2d<<<blocksPerGrid, threadsPerBlock, 0, stream>>>();

    gpuErrchk(cudaDeviceSynchronize());

    // Copy output : device -> host
    gpuErrchk(cudaMemcpyFromSymbol(h_output, d_output, O_MEM_SIZE));

    // Check output
    if (DEBUG) {
        convolution_cpu();
        checkOutput();
    }

    return 0;
}

__global__
void conv_2d() {
    unsigned int col = blockIdx.x;
    unsigned int row = blockIdx.y;

    unsigned int output_channel = threadIdx.z;
    

    __shared__ float input_cache[Ni][Ky][Kx];

    if (output_channel == 0) {
        for (int i = 0; i < Ni; i++)
            for (int y = 0; y < Ky; y++)
                for (int x = 0; x < Kx; x++)
                    input_cache[i][y][x] = d_input[i][row * StrideY + y][col * StrideX + x];
    } 

    __syncthreads();

    float sum = 0.0f;

    for (int i = 0; i < Ni; i++)
        for (int y = 0; y < Ky; y++)
            for (int x = 0; x < Kx; x++)
                sum += input_cache[i][y][x] * d_filters[output_channel][i][y][x];

    d_output[output_channel][row][col] = sum;


}

__host__
void randomizeFilters() {
    for (int yy = 0; yy < Ky; ++yy)
        for (int xx = 0; xx < Kx; ++xx)
            for (int nn = 0; nn < Nn; ++nn)
                for (int ni = 0; ni < Ni; ++ni)
                    h_filters[nn][ni][yy][xx] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
}

__host__
void randomizeInput() {
    for (int ni = 0; ni < Ni; ++ni)
        for (int yy = 0; yy < NyPad; ++yy)
            for (int xx = 0; xx < NxPad; ++xx)
                h_input[ni][yy][xx] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
}

__host__
void padInput() {
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



__host__
void convolution_cpu() {
    for (int nn = 0; nn < Nn; ++nn) {
        for (int oy = 0; oy < Oy; ++oy) {
            for (int ox = 0; ox < Ox; ++ox) {
                float sum = 0.0f;
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


__host__
void checkOutput() {
    for (int nn = 0; nn < Nn; ++nn) {
        for (int oy = 0; oy < Oy; ++oy) {
            for (int ox = 0; ox < Ox; ++ox) {
                if (std::abs(h_output[nn][oy][ox] - h_output_cpu[nn][oy][ox]) > 1e-3) {
                    printf("Mismatch at h_output[%d][%d][%d]: %f (CPU) vs %f (GPU)\n", nn, oy, ox, h_output_cpu[nn][oy][ox], h_output[nn][oy][ox]);
                    exit(1);
                }
            }
        }
    }
}
