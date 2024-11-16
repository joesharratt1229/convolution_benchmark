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
#include "utils/image_encoder/attention.cuh"

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

    constexpr int seq_len = 128;
    constexpr int output_dim = 256;
    constexpr int num_heads = 8;
    constexpr int embed_dim = output_dim / num_heads;

    floatT* query = new floatT[num_heads * seq_len * embed_dim];
    floatT* key = new floatT[num_heads * seq_len * embed_dim];
    floatT* value = new floatT[num_heads * seq_len * embed_dim];
    floatT* output = new floatT[num_heads * seq_len * embed_dim];
    floatT* output_cpu = new floatT[num_heads * seq_len * embed_dim];

    memset(output, 0, num_heads * seq_len * embed_dim * sizeof(floatT));

    randomizeInput(query, num_heads, seq_len, embed_dim);
    randomizeInput(key, num_heads, seq_len, embed_dim);
    randomizeInput(value, num_heads, seq_len, embed_dim);
    //flash_attention_kernel_wrapper<floatT, accFloatT, embed_dim, seq_len>(query, key, value, output, num_heads);
    scalable_flash_attention_kernel_wrapper<floatT, accFloatT, embed_dim, seq_len, num_heads>(query, key, value, output, num_heads);


    if (DEBUG) {
        multiHeadAttention_cpu<floatT, accFloatT>(query, key, value, output_cpu, seq_len, embed_dim, num_heads);
        for (int i = 0; i < 10; i++) {
            printf("%f %f\n", output[1000 + i], output_cpu[1000 + i]);
        }
        checkOutput(output, output_cpu, num_heads * seq_len * embed_dim);
    }

    /*model::NeckLayer<floatT, model::Nin1, model::Nin2, model::Nin3, model::Nin4> neck_layer;

    const char *filename = "model.bin";
    read_weights_from_file(filename, &neck_layer);

    XTensor<floatT>** x_input_arr = new XTensor<floatT>*[4];
    XTensor<floatT>** x_output_arr = new XTensor<floatT>*[4];
    XTensor<floatT>** pos_embeds_arr = new XTensor<floatT>*[4];

    XTensor<floatT>** x_output_arr_cpu = new XTensor<floatT>*[4];
    XTensor<floatT>** pos_embeds_arr_cpu = new XTensor<floatT>*[4];
    

    int input_channels[4] = {model::Nin1, model::Nin2, model::Nin3, model::Nin4};
    int output_size = Nx;

    for (int i = 0; i < 4; i++) {
        floatT* data_buf = new floatT[input_channels[i] * output_size * output_size];
        randomizeInput(data_buf, input_channels[i], output_size, output_size);

        Dimensions dims = {output_size, output_size, input_channels[i]};
        x_input_arr[i] = new XTensor<floatT>(data_buf, dims);

        free(data_buf);
        output_size = 2 * output_size;
    }

    output_size = Nx;

    for (int i = 0; i < 4; i++) {
        floatT* data_buf = new floatT[model::Nout * output_size * output_size];
        Dimensions dims = {output_size, output_size, model::Nout};
        x_output_arr[i] = new XTensor<floatT>(data_buf, dims);
        memset(x_output_arr[i]->get(), 0, model::Nout * output_size * output_size * sizeof(floatT));

        floatT* pos_embeds_buf = new floatT[model::Nout * output_size * output_size];
        pos_embeds_arr[i] = new XTensor<floatT>(pos_embeds_buf, dims);
        memset(pos_embeds_arr[i]->get(), 0, model::Nout * output_size * output_size * sizeof(floatT));

        x_output_arr_cpu[i] = new XTensor<floatT>(data_buf, dims);
        memset(x_output_arr_cpu[i]->get(), 0, model::Nout * output_size * output_size * sizeof(floatT));
        pos_embeds_arr_cpu[i] = new XTensor<floatT>(pos_embeds_buf, dims);
        memset(pos_embeds_arr_cpu[i]->get(), 0, model::Nout * output_size * output_size * sizeof(floatT));

        output_size = 2 * output_size;
    }


    image_encoder::template_conv_and_bilinear_resid_new<floatT, 1>(x_input_arr, x_output_arr, pos_embeds_arr, neck_layer);


    // Check output
    if (DEBUG) {
        for (int i = 0; i < 4; i++) {
            floatT* weight = neck_layer.get_layer_runtime(i)->conv[0][0][0];
            floatT* bias = neck_layer.get_layer_runtime(i)->bias;
            printf("Output dimensions: %d %d %d\n", x_output_arr_cpu[i]->get_dims().x_dimension, x_output_arr_cpu[i]->get_dims().y_dimension, x_output_arr_cpu[i]->get_dims().num_channels);

            convolution_cpu<floatT, 1>(x_input_arr[i]->get(), weight, bias, x_output_arr_cpu[i]->get(), x_input_arr[i]->get_dims(), x_output_arr_cpu[i]->get_dims());

            if (i > 1 ) {
                bilinear_interpolation_2x(x_output_arr_cpu[i-1]->get(), x_output_arr_cpu[i]->get(), x_output_arr_cpu[i-1]->get_dims(), x_output_arr_cpu[i]->get_dims());
                pos_embed_cpu(pos_embeds_arr_cpu[i]->get(), pos_embeds_arr_cpu[i]->get_dims());
            } else {
                pos_embed_cpu(pos_embeds_arr_cpu[i]->get(), pos_embeds_arr_cpu[i]->get_dims());
            }

            printf("Iteration number: %d\n", i);
            checkOutput(x_output_arr[i]->get(), x_output_arr_cpu[i]->get(), x_output_arr[i]->size());
            printf("Pos embeds\n");
            checkOutput(pos_embeds_arr[i]->get(), pos_embeds_arr_cpu[i]->get(), pos_embeds_arr[i]->size());
        }
    }

    return 0; */
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