#ifndef MODEL_CUH
#define MODEL_CUH

#include <tuple>
#include <stdexcept>

#include "utils/common.h"

namespace model {

constexpr int Nin1 = 1152;
constexpr int Nin2 = 576;
constexpr int Nin3 = 288;
constexpr int Nin4 = 144;

constexpr int Nout = 256;


template <typename T, int num_channels>
struct Layer {
    T conv[Nout][num_channels][1][1];
    T bias[Nout];
};

template <typename T, int... num_channels>
struct NeckLayer {
    std::tuple<Layer<T, num_channels>...> layers;
    static constexpr int num_layers = sizeof...(num_channels);

    template <size_t Index>
    auto& get_layer() {
        return std::get<Index>(layers);
    }

    static constexpr int size() {
        return num_layers;
    }

    struct ConvDimensions {
        int in_channels = 0;
        int out_channels = 0;
        int x_dimension = 0;
        int y_dimension = 0;
    } dims[num_layers];

    void set_dimensions(int index, int x_dim, int y_dim, int in_channels, int out_channels) {
        if (index < num_layers) {
            dims[index].in_channels = in_channels;
            dims[index].out_channels = out_channels;
            dims[index].x_dimension = x_dim;
            dims[index].y_dimension = y_dim;
        }
    }

    Layer<T, Nout>* get_layer_runtime(int index) {
        if (index < 0 || index >= num_layers) {
            throw std::out_of_range("Layer index out of bounds");
        }

        void* layer_ptr = nullptr;
        switch (index) {
            case 0: layer_ptr = &std::get<0>(layers); break;
            case 1: layer_ptr = &std::get<1>(layers); break;
            case 2: layer_ptr = &std::get<2>(layers); break;
            case 3: layer_ptr = &std::get<3>(layers); break;
            default: throw std::out_of_range("Layer index out of bounds");
        }
        return static_cast<Layer<T, Nout>*>(layer_ptr);
    }

};

}

#endif