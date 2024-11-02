#ifndef MODEL_CUH
#define MODEL_CUH

#include "utils/common.h"

namespace model {

constexpr int Nin1 = 1152;
constexpr int Nin2 = 576;
constexpr int Nin3 = 288;
constexpr int Nin4 = 144;

constexpr int Nout = 256;


typedef struct {
    floatT conv1[Nin1][Nout][1][1];
    floatT conv2[Nin2][Nout][1][1];
    floatT conv3[Nin3][Nout][1][1];
    floatT conv4[Nin4][Nout][1][1];

    floatT bias1[Nout];
    floatT bias2[Nout];
    floatT bias3[Nout];
    floatT bias4[Nout];
} NeckLayer;

}

#endif