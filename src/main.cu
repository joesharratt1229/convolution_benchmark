#include <string>
#include <cmath>

#include "utils/common.h" 
#include "utils/image_encoder/convolution.cuh"
#include "utils/image_encoder/upsample.cuh"
#include "utils/test_sam.cpp"

using namespace std;


template<typename T>
__host__ void randomizeFilters(T h_filters[Nn][Ni][Ky][Kx]);

template<typename T>
__host__ void randomizeInput(T h_input[Ni][NyPad][NxPad]);

template<typename T>
__host__ void padInput(T h_input[Ni][NyPad][NxPad]);

template<typename T>
__host__ void printParameters();

template<typename T>
__host__ void randomizeWindowEmbeddings(T h_window_embeds[Nn][WINDOW_EMBEDS][WINDOW_EMBEDS]);
template<typename T>
__host__
void randomizePosEmbeddings(T h_pos_embeds[Nn][POS_EMBEDS][POS_EMBEDS]);


int main(int argc, char **argv) {
    bool DEBUG = ((argc > 1) && (std::string(argv[1]) == "--debug"));


    static floatT h_input[Ni][NyPad][NxPad];
    static floatT h_convolution_output[Nn][Oy][Ox];
    static floatT h_output_bicubic[Nn][Oy][Ox];
    static floatT h_output_cpu[Nn][Oy][Ox];
    static floatT h_output_cpu_bicubic[Nn][Oy][Ox];
    static floatT h_filters[Nn][Ni][Ky][Kx]; 
    static floatT pos_embeds[Nn][POS_EMBEDS][POS_EMBEDS];
    static floatT h_window_embeds[Nn][WINDOW_EMBEDS][WINDOW_EMBEDS];

    dims input_dims = {POS_EMBEDS, POS_EMBEDS, Nn};
    dims output_dims = {Ox, Oy, Nn};
    randomizeFilters(h_filters);
    randomizeInput(h_input);
    padInput(h_input);
    randomizePosEmbeddings(pos_embeds);
    randomizeWindowEmbeddings(h_window_embeds);
    image_encoder::template_conv_2d<floatT, 16>(h_input, h_filters, h_convolution_output);
    /*image_encoder::template_bicubic_upsample_and_window_embed<floatT, POS_EMBEDS, Nn, Oy, Ox, 16, WINDOW_EMBEDS>(pos_embeds, 
                                                                                                 h_output_bicubic, 
                                                                                                 h_convolution_output,
                                                                                                 h_window_embeds, 
                                                                                                 input_dims, 
                                                                                                 output_dims);*/

    // Check output
    if (DEBUG) {
        convolution_cpu(h_input, h_filters, h_output_cpu);
        //bicubic_convolution_cpu(pos_embeds, Oy, Ox, h_output_cpu_bicubic);
        checkOutput(&h_convolution_output[0][0][0], &h_output_cpu[0][0][0], Ox * Oy * Nn);
        //checkOutput(&h_output_bicubic[0][0][0], &h_output_cpu_bicubic[0][0][0], Ox * Oy * Nn);
    } 

    return 0;
} 

template<typename T>
__host__ void randomizeWindowEmbeddings(T h_window_embeds[Nn][WINDOW_EMBEDS][WINDOW_EMBEDS]) {
    for (int nn = 0; nn < Nn; ++nn)
        for (int yy = 0; yy < WINDOW_EMBEDS; ++yy)
            for (int xx = 0; xx < WINDOW_EMBEDS; ++xx)
                h_window_embeds[nn][yy][xx] = static_cast<T>(static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f);
}


template<typename T>
__host__
void randomizePosEmbeddings(T h_pos_embeds[Nn][POS_EMBEDS][POS_EMBEDS]) {
    for (int nn = 0; nn < Nn; ++nn)
        for (int yy = 0; yy < POS_EMBEDS; ++yy)
            for (int xx = 0; xx < POS_EMBEDS; ++xx)
                h_pos_embeds[nn][yy][xx] = static_cast<T>(static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f);
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


