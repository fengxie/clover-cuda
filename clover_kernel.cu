
#include <cstdio>

#include <cuda.h>
#include <math_constants.h>

#include <math.h>

#define CUDART_PI_F 3.141592654f

__global__ void clover_kernel(uchar4 *image, float scale, size_t stride, const dim3 center)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    uchar4 *ptr = image + ty * stride + tx;

    float x = tx - (int)center.x;
    float y = (int)center.y - ty;

    float radius = sqrt(x * x + y * y);
    float theta  = atan2(y, x) + CUDART_PI_F / 6.0;
    float bound = scale * abs(sin(2.0f * theta) + sin(6.0f * theta) / 4.0f);

    uchar4 rgba = make_uchar4(255, 0, 0, 255);

    if (radius < bound && radius > 0.5 * bound) {
        rgba.x = 0;
        rgba.y = 255;
        rgba.z = 0;
        rgba.w = 255;
    } else if (radius < 0.5 * bound) {
        rgba.x = 128;
        rgba.y = 255;
        rgba.z = 128;
        rgba.w = 255;
    }

    *ptr = rgba;
}
