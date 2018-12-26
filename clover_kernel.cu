
#include <cstdio>

#include <cuda.h>
#include <math_constants.h>

#include <math.h>

#define CUDART_PI_F 3.141592654f

template<int AA_X_ = 2,
         int AA_Y_ = 2>
__global__ void clover_kernel(uchar4 *image, uchar4 *background, float scale, size_t stride, const dim3 center)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    uchar4 *ptr = image + ty * stride + tx;
    uchar4 *inptr = background + ty * stride + tx;

    uchar4 rgba_out = make_uchar4(0, 0, 0, 0);
    for (int i = 0; i < AA_Y_; i++) {
        for (int j = 0; j < AA_X_; j++) {
            float x = tx - (int)center.x + j * 1.0f / AA_X_;
            float y = (int)center.y - ty + i * 1.0f / AA_Y_;

            float radius = sqrt(x * x + y * y);
            float theta  = atan2(y, x) + CUDART_PI_F / 6.0;
            float bound = scale * abs(sin(2.0f * theta) + sin(6.0f * theta) / 4.0f);

            uchar4 rgba = make_uchar4(0, 0, 0, 255);

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

            rgba_out.x += rgba.x / (AA_X_ * AA_Y_);
            rgba_out.y += rgba.y / (AA_X_ * AA_Y_);
            rgba_out.z += rgba.z / (AA_X_ * AA_Y_);
            rgba_out.w += rgba.w / (AA_X_ * AA_Y_);
        }
    }

#if 0
    float f = 0.8;
    float s = f + (1 - f) * (1.0f - (inptr[0].x + inptr[0].y + inptr[0].z + inptr[0].w) / 1024.0f);
#else
    float s = 1.0f;
#endif
    rgba_out.x = s * rgba_out.x;
    rgba_out.y = s * rgba_out.y;
    rgba_out.z = s * rgba_out.z;
    rgba_out.w = s * rgba_out.w;

    *ptr = rgba_out;
}

template __global__ void clover_kernel<2, 2>(uchar4 *image, uchar4 *background, float scale, size_t stride, const dim3 center);
template __global__ void clover_kernel<4, 4>(uchar4 *image, uchar4 *background, float scale, size_t stride, const dim3 center);

template<int tile_x = 4,
         int tile_y = 4,
         int filter_size = 3>
__global__ void convolve(uchar4 *image, float scale, unsigned char kernel[filter_size][filter_size], size_t stride)
{
}

