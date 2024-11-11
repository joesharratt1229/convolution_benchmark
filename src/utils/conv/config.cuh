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

struct Dimensions {
    int x_dimension;
    int y_dimension;
    int num_channels;
};


template <typename T>
class XTensor {
    private:
        T* data;
        Dimensions dims;
        cudaStream_t stream;
        bool is_device_tensor;

    public:
        XTensor(const T* data, const Dimensions dims) : dims(dims)
        {
            this->data = (T*)malloc(size() * sizeof(T));
            memcpy(this->data, data, size() * sizeof(T));
        }

        XTensor(const Dimensions dims) : dims(dims) {
            this->data = (T*)malloc(size() * sizeof(T));
        }

        XTensor(const XTensor<T>& other, 
                const Dimensions dims, 
                cudaStream_t stream = 0, 
                bool is_device_tensor = true, 
                bool is_async = false) : dims(dims), stream(stream), is_device_tensor(is_device_tensor)
        {
            if (is_device_tensor && !is_async) {
                gpuErrchk(cudaMalloc((void**)&this->data, size() * sizeof(T)));
                gpuErrchk(cudaMemcpy(this->data, other.data, size() * sizeof(T), cudaMemcpyHostToDevice));
            } else if (is_device_tensor && is_async) {
                gpuErrchk(cudaMalloc((void**)&this->data, size() * sizeof(T)));
                gpuErrchk(cudaMemcpyAsync(this->data, other.data, size() * sizeof(T), cudaMemcpyHostToDevice, stream));
            } else {
                this->data = (T*)malloc(size() * sizeof(T));
                memcpy(this->data, other.data, size() * sizeof(T));
            }
        }

        ~XTensor() {
            cleanup();
        }

        size_t size() const noexcept {
            return static_cast<size_t>(dims.x_dimension) * 
                   static_cast<size_t>(dims.y_dimension) * 
                   static_cast<size_t>(dims.num_channels);
        }

        Dimensions get_dims() const {
            return dims;
        }

        int x_dim() const {
            return dims.x_dimension;
        }

        int y_dim() const {
            return dims.y_dimension;
        }

        int channels() const {
            return dims.num_channels;
        }

        T* get() {
            return data;
        }

private:
    void cleanup() {
        if (data) {
            if (is_device_tensor) {
                cudaFree(data);
            } else {
                free(data);
            }
            data = nullptr;
        }
    }
};



/*template <typename T>
struct x_tensor {
    T* data[NUM_INPUTS];  // Array of pointers

    Dimensions dims[NUM_INPUTS];

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
*/

#endif
