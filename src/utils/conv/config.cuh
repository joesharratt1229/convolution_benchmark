#ifndef CONV_CONFIG_H
#define CONV_CONFIG_H

#include <type_traits>

template<int KernelSize>
struct KernelTraits ;

#define NUM_INPUTS 4

template<int KernelSize>
struct TileConfig {
    using Traits = KernelTraits<KernelSize>;
    
    static constexpr int KERNEL_SIZE = KernelSize;
    static constexpr int TILE_SIZE = Traits::TILE_SIZE;
    static constexpr int CHANNEL_TILE_SIZE = Traits::CHANNEL_TILE_SIZE;
    static constexpr int STRIDE = Traits::STRIDE;
    static constexpr int INPUT_TILE_SIZE = STRIDE * TILE_SIZE + KERNEL_SIZE - 1;
    
    static_assert(KERNEL_SIZE % 2 == 1, "Kernel size must be odd");
    static_assert(TILE_SIZE > 0, "Tile size must be positive");
    static_assert(CHANNEL_TILE_SIZE > 0, "Channel tile size must be positive");
    static_assert(STRIDE > 0, "Stride must be positive");
};


template<>
struct KernelTraits<3> {
    static constexpr int TILE_SIZE = 16;
    static constexpr int CHANNEL_TILE_SIZE = 1;
    static constexpr int STRIDE = 1;
};


template<>
struct KernelTraits<1> {
    static constexpr int TILE_SIZE = 32;
    static constexpr int CHANNEL_TILE_SIZE = 1;
    static constexpr int STRIDE = 1;
};

template<>
struct KernelTraits<7> {
    static constexpr int TILE_SIZE = 8;
    static constexpr int CHANNEL_TILE_SIZE = 8;
    static constexpr int STRIDE = 4;
};

template<int kernel_size>
constexpr TileConfig<kernel_size> get_tile_config() {
    return TileConfig<kernel_size>{};
}


template <typename T>
struct x_tensor {
    T* data[NUM_INPUTS];  // Array of pointers

    struct Dimensions {
        int x_dimension;
        int y_dimension;
        int num_channels;
    } dims[NUM_INPUTS];

    x_tensor() = default;


    void set_dimensions(int index, int x_dim, int y_dim, int channels) {
        if (index < NUM_INPUTS) {
            dims[index].x_dimension = x_dim;
            dims[index].y_dimension = y_dim;
            dims[index].num_channels = channels;
        }
    }

    // Getter methods for dimensions of a specific array
    int x_dim(int index) const { 
        return dims[index].x_dimension; 
    }
    
    int y_dim(int index) const { 
        return dims[index].y_dimension; 
    }
    
    int channels(int index) const { 
        return dims[index].num_channels; 
    }

    T* get(int index) {
        return data[index];
    }

    const T* get(int index) const {  // Added const version for read-only access
        return data[index];
    }

    static constexpr int size() {
        return NUM_INPUTS;
    }
};

#endif
