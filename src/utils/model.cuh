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
    T conv[num_channels][Nout][1][1];
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