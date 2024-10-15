#include <cuda_fp16.h>
#include <cuda_bf16.h>


#if defined(ENABLE_FP32)
typedef float floatT;
#elif defined(ENABLE_FP16)
typedef half floatT;
#elif defined(ENABLE_BP16)
typedef __nv_bfloat16 floatT;
#endif

