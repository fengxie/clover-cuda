
#include <cuda.h>
#include <math_constants.h>

#define CUDART_PI_F 3.141592654f
#include <math.h>

__global__ void clover_kernel(uchar4 *image, size_t stride, const dim2 center)
{
    float x = blockIdx.x * blockDim.x + threadIdx.x - center.x;
    float y = blockIdx.y * blockDim.y + threadIdx.y - center.y;

    uchar4 *ptr = image + (blockIdx.y * blockDim.y + threadIdx.y) * stride + blockIdx.x * blockDim.x + threadIdx.x;

    float theta_off = 0;

    float radius = sqrt(x * x + y * y)
    float theta  = atan2(y, x)

    float bound = sin(2 * theta) + sin(6 * theta) / 4.0
    uchar4 rgba = make_uchar4(0, 0, 0, 0);
    if (radius < bound) {
        rgba.x = 0;
        rgba.y = 255;
        rgba.z = 0;
        rgba.w = 255;
    }

    *ptr = rgba;
}