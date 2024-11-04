#include <string>
#include <cmath>
#include <stdint.h>
#include <cassert>

#include "utils/common.h" 
#include "utils/image_encoder/convolution.cuh"
#include "utils/image_encoder/upsample.cuh"
#include "utils/image_encoder/posEmbedding.cuh"
#include "utils/image_encoder/neck.cuh"
#include "utils/model.cuh"
#include "utils/test_sam.cpp"
#include "utils/conv/config.cuh"

using namespace std;


void read_weights_from_file(const char *filename, 
                            model::NeckLayer<floatT, model::Nin1, model::Nin2, model::Nin3, model::Nin4>* neck_layer);

template<typename T, int NnDim, int NiDim, int KyDim, int KxDim>
__host__ void randomizeFilters(T h_filters[NnDim][NiDim][KyDim][KxDim]);

template<typename T> 
__host__ void randomizeInput(T* h_input, int NiDim, int NyDim, int NxDim);

template<typename T>
__host__ void padInput(T h_input[Ni][NyPad][NxPad]);

template<typename T>
__host__ void printParameters();

template<typename T>
__host__ void randomizeWindowEmbeddings(T h_window_embeds[Nn][WINDOW_EMBEDS][WINDOW_EMBEDS]);

template<typename T, int num_channels>
__host__ void randomiseConvBias(T h_bias[num_channels]);

template<typename T, int NnDim, int x_dim, int y_dim>
void randomizePosEmbeddings(T h_pos_embeds[NnDim][x_dim][y_dim]);


int main(int argc, char **argv) {
    bool DEBUG = ((argc > 1) && (std::string(argv[1]) == "--debug"));

    model::NeckLayer<floatT, model::Nin1, model::Nin2, model::Nin3, model::Nin4> neck_layer;

    const char *filename = "model.bin";
    read_weights_from_file(filename, &neck_layer);

    x_tensor<floatT> x_input;
    x_tensor<floatT> x_output;
    x_tensor<floatT> pos_embeds;

    int input_channels[4] = {model::Nin1, model::Nin2, model::Nin3, model::Nin4};

    for (int i = 0; i < x_input.size(); i++) {
        int output_size = Nx;

        x_input.data[i] = (floatT*)malloc(model::Nin1 * output_size * output_size * sizeof(floatT));

        if (x_input.data[i] == NULL) {
            printf("Error allocating memory for x_input.data[%d]\n", i);
            exit(1);
        }
        
        x_input.set_dimensions(i, output_size, output_size, input_channels[i]);
        randomizeInput(x_input.data[i], input_channels[i], output_size, output_size);
        output_size *= 2;
    }

    for (int i = 0; i < x_output.size(); i++) {
        int output_size = Nx;
        x_output.data[i] = (floatT*)malloc(model::Nout * output_size * output_size * sizeof(floatT));
        
        if (x_output.data[i] == NULL) {
            printf("Error allocating memory for x_output.data[%d]\n", i);
            exit(1);
        }
        x_output.set_dimensions(i, output_size, output_size, model::Nout);
        memset(x_output.data[i], 0, model::Nout * output_size * output_size * sizeof(floatT));
    

        pos_embeds.data[i] = (floatT*)malloc(model::Nout * output_size * output_size * sizeof(floatT));

        if (pos_embeds.data[i] == NULL) {
            printf("Error allocating memory for pos_embeds.data[%d]\n", i);
            exit(1);
        }
        pos_embeds.set_dimensions(i, output_size, output_size, model::Nout);
        memset(pos_embeds.data[i], 0, model::Nout * output_size * output_size * sizeof(floatT));

    }

    image_encoder::template_conv_and_bilinear_resid_new<floatT, 1>(x_input, x_output, pos_embeds, neck_layer);




    /*static floatT h_input[Ni][NyPad][NxPad];
    static floatT h_input_1x1[N1x1][Ny][Nx];
    static floatT h_previous_input[Nn][Ny/2][Nx/2];

    static floatT h_convolution_output[Nn][Oy][Ox];
    static floatT h_output_cpu[Nn][Oy][Ox];
    static floatT h_output_bicubic[Nn][Oy][Ox];
    static floatT h_output_cpu_bicubic[Nn][Oy][Ox];

    static floatT h_filters_7x7[Nn][Ni][Ky][Kx]; 

    static floatT h_1x1_output[Nn][Ny][Nx];
    static floatT h_1x1_output_cpu[Nn][Ny][Nx];
    static floatT h_backbone_output[Nn][Ny][Nx];
    static floatT h_backbone_output_cpu[Nn][Ny][Nx];
    static floatT h_filters_1x1[Nn][N1x1][1][1];
    static floatT h_bias_1x1[Nn];

    static floatT pos_embeds[Nn][POS_EMBEDS][POS_EMBEDS];
    static floatT h_window_embeds[Nn][WINDOW_EMBEDS][WINDOW_EMBEDS];


    randomizeFilters<floatT, Nn, Ni, Ky, Kx>(h_filters_7x7);  
    randomizeFilters<floatT, Nn, N1x1, 1, 1>(h_filters_1x1);
    randomizeInput<floatT, Ni, NyPad, NxPad>(h_input);
    randomizeInput<floatT, N1x1, Ny, Nx>(h_input_1x1);
    randomizeInput<floatT, Nn, Ny/2, Nx/2>(h_previous_input);
    randomiseConvBias<floatT, Nn>(h_bias_1x1);
    padInput(h_input);
    randomizePosEmbeddings<floatT, Nn, POS_EMBEDS, POS_EMBEDS>(pos_embeds);
    randomizeWindowEmbeddings(h_window_embeds);

    //printf("SM count: %d\n", getSMCount());

    image_encoder::template_conv_2d<floatT, 7, Ni, Nn, image_encoder::ConvImplementation::Shared>(&h_input[0][0][0], &h_convolution_output[0][0][0], h_filters_7x7, h_bias_1x1);
    image_encoder::template_bicubic_upsample_and_window_embed<floatT>(&pos_embeds[0][0][0], 
                                                                     &h_output_bicubic[0][0][0], 
                                                                     &h_convolution_output[0][0][0],
                                                                     &h_window_embeds[0][0][0]);
    

    image_encoder::template_conv_and_bilinear_resid<floatT, 1>(&h_input_1x1[0][0][0],
                                                               &h_previous_input[0][0][0],
                                                               &h_1x1_output[0][0][0],
                                                               &h_backbone_output[0][0][0],
                                                               h_filters_1x1,
                                                               h_bias_1x1);


    // Check output
    if (DEBUG) {
        convolution_cpu<floatT, Ni, NyPad, NxPad, Nn, Oy, Ox, 7>(h_input, h_filters_7x7, h_bias_1x1, h_output_cpu);
        bicubic_convolution_cpu<floatT, Nn, POS_EMBEDS, POS_EMBEDS, Oy, Ox>(pos_embeds, Oy, Ox, h_output_cpu_bicubic);
        convolution_cpu<floatT, N1x1, Ny, Nx, Nn, Ny, Nx, 1>(h_input_1x1, h_filters_1x1, h_bias_1x1, h_1x1_output_cpu);
        bilinear_interpolation_2x<floatT, Nn, Ny, Nx>(h_previous_input, h_backbone_output_cpu, h_1x1_output_cpu);
        checkOutput(&h_convolution_output[0][0][0], &h_output_cpu[0][0][0], Ox * Oy * Nn);
        checkOutput(&h_output_bicubic[0][0][0], &h_output_cpu_bicubic[0][0][0], Ox * Oy * Nn);
        checkOutput(&h_backbone_output[0][0][0], &h_backbone_output_cpu[0][0][0], Nx * Ny * Nn);
        checkOutput(&h_1x1_output[0][0][0], &h_1x1_output_cpu[0][0][0], Nx * Ny * Nn);
    } */

    return 0;
} 

template<typename T>
__host__ void randomizeWindowEmbeddings(T h_window_embeds[Nn][WINDOW_EMBEDS][WINDOW_EMBEDS]) {
    for (int nn = 0; nn < Nn; ++nn)
        for (int yy = 0; yy < WINDOW_EMBEDS; ++yy)
            for (int xx = 0; xx < WINDOW_EMBEDS; ++xx)
                h_window_embeds[nn][yy][xx] = static_cast<T>(static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f);
}

template<typename T, int num_channels>
__host__ void randomiseConvBias(T h_bias[num_channels]) {
    for (int nn = 0; nn < num_channels; ++nn)
        h_bias[nn] = static_cast<T>(static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f);
}

template<typename T, int Num_channels, int x_dim, int y_dim>
__host__
void randomizePosEmbeddings(T h_pos_embeds[Num_channels][x_dim][y_dim]) {
    for (int nn = 0; nn < Num_channels; ++nn)
        for (int yy = 0; yy < y_dim; ++yy)
            for (int xx = 0; xx < x_dim; ++xx)
                h_pos_embeds[nn][yy][xx] = static_cast<T>(static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f);
}


template<typename T, int NnDim, int NiDim, int KyDim, int KxDim>
__host__
void randomizeFilters(T h_filters[NnDim][NiDim][KyDim][KxDim]) {
    for (int yy = 0; yy < Ky; ++yy)
        for (int xx = 0; xx < Kx; ++xx)
            for (int nn = 0; nn < Nn; ++nn)
                for (int ni = 0; ni < Ni; ++ni)
                    h_filters[nn][ni][yy][xx] = static_cast<T>(static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f);
}


template<typename T>
__host__
void randomizeInput(T* h_input, int NiDim, int NyDim, int NxDim) {
    for (int ni = 0; ni < NiDim; ++ni)
        for (int yy = 0; yy < NyDim; ++yy)
            for (int xx = 0; xx < NxDim; ++xx) {
                int idx = ni * (NyDim * NxDim) + yy * NxDim + xx;
                h_input[idx] = static_cast<T>(static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f);
            }
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


void read_weights_from_file(const char *filename, 
                            model::NeckLayer<floatT, model::Nin1, model::Nin2, model::Nin3, model::Nin4>* neck_layer) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error opening file %s\n", filename);
        exit(1);
    }

    int32_t num_layers;

    if(fread(&num_layers, sizeof(int32_t), 1, file) != 1) {
        printf("Error reading number of layers from file %s\n", filename);
        exit(1);
    }

    assert(num_layers == neck_layer->size() && "Invalid number of layers");

    for (int i = 0; i < neck_layer->size(); i++) {
        auto* layer = neck_layer->get_layer_runtime(i);
        int32_t dims[4];
        if(fread(&dims, sizeof(int32_t) * 4, 1, file) != 1) {
            printf("Error reading dimensions from file %s\n", filename);
            exit(1);
        }
        printf("Layer %d dimensions: %d %d %d %d\n", i, dims[0], dims[1], dims[2], dims[3]);

        floatT *temp_buf = (floatT*)malloc(dims[0] * dims[1] * sizeof(floatT));

        if(fread(temp_buf, sizeof(floatT), dims[0] * dims[1] , file) != dims[0] * dims[1]) {
            printf("Error reading weights from file %s\n", filename);
            exit(1);
        }

        for (int j = 0; j < dims[0]; j++) {
            for (int k = 0; k < dims[1]; k++) {
                layer->conv[j][k][0][0] = temp_buf[j * dims[1] + k];
            }
        }

        neck_layer->set_dimensions(i, dims[2], dims[3], dims[1], dims[0]);

        floatT *temp_buf_bias = (floatT*)malloc((dims[0]+1) * sizeof(floatT));

        if(fread(temp_buf_bias, sizeof(floatT), dims[0]+1, file) != dims[0]+1) {
            printf("Error reading biases from file %s\n", filename);
            exit(1);
        }

        for (int j = 0; j < dims[0]+1; j++) {
            layer->bias[j] = temp_buf_bias[j+1];
        }

        free(temp_buf);
        free(temp_buf_bias);
    }

    fclose(file);
}