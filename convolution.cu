#include <string>
#include <cmath>

#define Pad 3
#define StrideX 4
#define StrideY 4
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
#define TILE_SIZE 8
#define CHANNEL_SIZE 16
#define INPUT_TILE_X (TILE_SIZE*StrideX + Kx - 1)
#define INPUT_TILE_Y (TILE_SIZE*StrideY + Ky - 1)

using namespace std;

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void conv_2d(float d_input[Ni][NyPad][NxPad], float d_filters[Nn][Ni][Ky][Kx], float d_output[Nn][Oy][Ox]);

__host__ void randomizeFilters(float h_filters[Nn][Ni][Ky][Kx]);
__host__ void randomizeInput(float h_input[Ni][NyPad][NxPad]);
__host__ void padInput(float h_input[Ni][NyPad][NxPad]);
__host__ void printParameters();
__host__ void convolution_cpu(float h_input[Ni][NyPad][NxPad], float h_filters[Nn][Ni][Ky][Kx], float h_output_cpu[Nn][Oy][Ox]);
__host__ void checkOutput(float h_output[Nn][Oy][Ox], float h_output_cpu[Nn][Oy][Ox]);


int main(int argc, char **argv) {
    bool DEBUG = ((argc > 1) && (std::string(argv[1]) == "--debug"));

    unsigned int Ox2 = (Ox + 1) / 2;

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE, CHANNEL_SIZE);
    dim3 blocksPerGrid((Ox2 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (Oy + threadsPerBlock.y - 1) / threadsPerBlock.y,
                       (Nn + threadsPerBlock.z - 1) / threadsPerBlock.z);


    static float h_input[Ni][NyPad][NxPad];
    static float h_output[Nn][Oy][Ox];
    static float h_output_cpu[Nn][Oy][Ox];
    static float h_filters[Nn][Ni][Ky][Kx]; 

    float (*d_input)[NyPad][NxPad];
    float (*d_output)[Oy][Ox];
    float (*d_filters)[Ni][Ky][Kx];


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
        checkOutput(h_output, h_output_cpu);
    } 

    return 0;
} 



__global__
void conv_2d(float d_input[Ni][NyPad][NxPad], float d_filters[Nn][Ni][Ky][Kx], float d_output[Nn][Oy][Ox]) {
    unsigned int col = 2*(blockIdx.x * TILE_SIZE + threadIdx.x);
    unsigned int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    unsigned int output_channel = blockIdx.z * CHANNEL_SIZE + threadIdx.z;

    __shared__ float input_cache[Ni][INPUT_TILE_Y][INPUT_TILE_X*2];

    if (threadIdx.z < Ni && StrideY * threadIdx.y < INPUT_TILE_Y && StrideX * threadIdx.x < INPUT_TILE_X) {
        if (threadIdx.y < TILE_SIZE - 1)
            for (int y = 0; y < StrideY; y++)
                if (threadIdx.x < TILE_SIZE - 1)
                    for (int x = 0; x < 2* StrideX; x++) 
                        input_cache[threadIdx.z][StrideY * threadIdx.y + y][2*StrideX * threadIdx.x + x] = d_input[threadIdx.z][row*StrideY + y][col*StrideX + x];
                else
                    for (int x = 0; x < 2* StrideX + Kx - 1; x++) 
                        input_cache[threadIdx.z][StrideY * threadIdx.y + y][2*StrideX * threadIdx.x + x] = d_input[threadIdx.z][row*StrideY + y][col*StrideX + x];
        else 
            for (int y = 0; y < StrideY + Ky - 1; y++)
                if (threadIdx.x < TILE_SIZE - 1)
                    for (int x = 0; x < 2* StrideX; x++) 
                        input_cache[threadIdx.z][StrideY * threadIdx.y + y][2*StrideX * threadIdx.x + x] = d_input[threadIdx.z][row*StrideY + y][col*StrideX + x];
                else
                    for (int x = 0; x < 2* StrideX + Kx - 1; x++) 
                        input_cache[threadIdx.z][StrideY * threadIdx.y + y][2*StrideX * threadIdx.x + x] = d_input[threadIdx.z][row*StrideY + y][col*StrideX + x];
    }

    __syncthreads();

    float sum1 = 0.0f;
    float sum2 = 0.0f;

    if (row < Oy && output_channel < Nn) {
        #pragma unroll
        for (int i = 0; i < Ni; i++)
            #pragma unroll
            for (int y = 0; y < Ky; y++)
                #pragma unroll
                for (int x = 0; x < Kx; x++) {
                    float filter_val = d_filters[output_channel][i][y][x];
                    sum1 += input_cache[i][threadIdx.y * StrideY + y][(2*threadIdx.x) * StrideX + x] * filter_val;
                    sum2 += input_cache[i][threadIdx.y * StrideY + y][(2*threadIdx.x+1) * StrideX + x] * filter_val;
                }
        
        if (col < Ox) {
            d_output[output_channel][row][col] = sum1;
        }
        if (col+1 < Ox) {
            d_output[output_channel][row][col+1] = sum2;
        }
    }
}

__host__
void randomizeFilters(float h_filters[Nn][Ni][Ky][Kx]) {
    for (int yy = 0; yy < Ky; ++yy)
        for (int xx = 0; xx < Kx; ++xx)
            for (int nn = 0; nn < Nn; ++nn)
                for (int ni = 0; ni < Ni; ++ni)
                    h_filters[nn][ni][yy][xx] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
}

__host__
void randomizeInput(float h_input[Ni][NyPad][NxPad]) {
    for (int ni = 0; ni < Ni; ++ni)
        for (int yy = 0; yy < NyPad; ++yy)
            for (int xx = 0; xx < NxPad; ++xx)
                h_input[ni][yy][xx] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
}

__host__
void padInput(float h_input[Ni][NyPad][NxPad]) {
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
void convolution_cpu(float h_input[Ni][NyPad][NxPad], float h_filters[Nn][Ni][Ky][Kx], float h_output_cpu[Nn][Oy][Ox]) {
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
void checkOutput(float h_output[Nn][Oy][Ox], float h_output_cpu[Nn][Oy][Ox]) {
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