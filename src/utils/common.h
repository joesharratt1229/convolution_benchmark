#ifndef COMMON_H
#define COMMON_H

#include <cuda_fp16.h>
#include <cuda_bf16.h>


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
#define I_MEM_SIZE (I_SIZE * sizeof(floatT))
#define O_MEM_SIZE (O_SIZE * sizeof(floatT))
#define F_MEM_SIZE (F_SIZE * sizeof(floatT))
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define POS_EMBEDS 14
#define nStreams 8
#define WINDOW_EMBEDS 8

#define EPSILON 1e-6
#define TEMPERATURE 10000
#define SCALE 2*M_PI

typedef struct {
    int width;
    int height;
    int channel;
} dims;


#if defined(ENABLE_FP32)
typedef float floatT;
#elif defined(ENABLE_FP16)
typedef half floatT;
#elif defined(ENABLE_BP16)
typedef __nv_bfloat16 floatT;
#endif

typedef float accFloatT;


#endif