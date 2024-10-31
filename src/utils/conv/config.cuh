#ifndef CONV_CONFIG_H
#define CONV_CONFIG_H

#include <type_traits>

template<int KernelSize>
struct KernelTraits ;


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
struct KernelTraits<7> {
    static constexpr int TILE_SIZE = 8;
    static constexpr int CHANNEL_TILE_SIZE = 8;
    static constexpr int STRIDE = 4;
};

template<int kernel_size>
constexpr TileConfig<kernel_size> get_tile_config() {
    return TileConfig<kernel_size>{};
}

#endif
