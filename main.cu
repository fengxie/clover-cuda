
#include <cinttypes>

#include <cuda.h>

#include <png.h>

#include "util.h"

template<int AA_X_ = 2,
         int AA_Y_ = 2>
__global__ void clover_kernel(uchar4 *image, uchar4 *background, float scale, size_t stride, const dim3 center);
extern int write_png(const char *filename, int width, int height, unsigned char *buffer, char* title);
extern int read_png(const char* filename, int width, int height, unsigned char *buffer);

int main(const int argc, const char *argv[])
{
    unsigned char *input;
    unsigned char *device_input;
    unsigned char *output;
    unsigned char *device_output;

    const size_t height = 1024;
    const size_t width  = 1024;
    const size_t size   = 4 * height * width;
    input  = new unsigned char[size];
    output = new unsigned char[size];

    read_png("clover.png", width, height, input);

    CUDA_ASSERT(cudaMalloc((void **)&device_input, size * sizeof(unsigned char)), "");
    CUDA_ASSERT(cudaMalloc((void **)&device_output, size * sizeof(unsigned char)), "");

    CUDA_ASSERT(cudaMemcpy(device_input, input, size * sizeof(unsigned char), cudaMemcpyHostToDevice), "");
    CUDA_ASSERT(cudaMemcpy(device_output, output, size * sizeof(unsigned char), cudaMemcpyHostToDevice), "");

    dim3 threads(16, 16, 1);
    dim3 blocks(width / threads.x, height / threads.y, 1);
    dim3 center(width / 2, height / 2, 1);
    clover_kernel<4, 4><<<blocks, threads>>>((uchar4 *)device_output, (uchar4 *)device_input, 512.0f, width, center);

    CUDA_ASSERT(cudaMemcpy(output, device_output, size * sizeof(unsigned char), cudaMemcpyDeviceToHost), "");
    write_png("clover.png", width, height, output, "Clover");

    CUDA_ASSERT(cudaFree(device_input), "");
    CUDA_ASSERT(cudaFree(device_output), "");

    delete[] input;
    delete[] output;

    return 0;
}

int read_png(const char* filename, int width, int height, unsigned char *buffer)
{
    int code = 0;
    FILE *fp = NULL;
    png_structp png_ptr = NULL;

    // Open file for writing (binary mode)
    fp = fopen(filename, "r");
    if (fp == NULL) {
        fprintf(stderr, "Could not open file %s for writing\n", filename);
        code = 1;
        goto finalise;
    }

    // Initialize write structure
    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (png_ptr == NULL) {
        fprintf(stderr, "Could not allocate write struct\n");
        code = 1;
        goto finalise;
    }

    /* Allocate/initialize the memory for image information.  REQUIRED. */
    png_infop info_ptr;
    info_ptr = png_create_info_struct(png_ptr);
    if (info_ptr == NULL) {
        fclose(fp);
        png_destroy_read_struct(&png_ptr, (png_infopp)NULL, (png_infopp)NULL);
        goto finalise;
    }


    png_init_io(png_ptr, fp);
    png_read_png(png_ptr, info_ptr, NULL, NULL);

    png_bytep *rows;
    rows = png_get_rows(png_ptr, info_ptr);

    for (int y = 0 ; y < height ; y++) {
        for (int x = 0; x < width; x++) {
            buffer[y * width * 4 + 4 * x + 0] = rows[y][4 * x + 0];
            buffer[y * width * 4 + 4 * x + 1] = rows[y][4 * x + 1];
            buffer[y * width * 4 + 4 * x + 2] = rows[y][4 * x + 2];
            buffer[y * width * 4 + 4 * x + 3] = rows[y][4 * x + 3];
        }
    }

finalise:
    if (fp != NULL) fclose(fp);
    if (png_ptr != NULL) png_destroy_read_struct(&png_ptr, (png_infopp)NULL, (png_infopp)NULL);

    return code;
}

int write_png(const char* filename, int width, int height, unsigned char *buffer, char* title)
{
   int code = 0;
   FILE *fp = NULL;
   png_structp png_ptr = NULL;
   png_infop info_ptr = NULL;

   // Open file for writing (binary mode)
   fp = fopen(filename, "wb");
   if (fp == NULL) {
      fprintf(stderr, "Could not open file %s for writing\n", filename);
      code = 1;
      goto finalise;
   }

   // Initialize write structure
   png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
   if (png_ptr == NULL) {
      fprintf(stderr, "Could not allocate write struct\n");
      code = 1;
      goto finalise;
   }

   // Initialize info structure
   info_ptr = png_create_info_struct(png_ptr);
   if (info_ptr == NULL) {
      fprintf(stderr, "Could not allocate info struct\n");
      code = 1;
      goto finalise;
   }

   // Setup Exception handling
   if (setjmp(png_jmpbuf(png_ptr))) {
      fprintf(stderr, "Error during png creation\n");
      code = 1;
      goto finalise;
   }

   png_init_io(png_ptr, fp);

   // Write header (8 bit colour depth)
   png_set_IHDR(png_ptr, info_ptr, width, height,
         8, PNG_COLOR_TYPE_RGBA, PNG_INTERLACE_NONE,
         PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

   // Set title
   if (title != NULL) {
      png_text title_text;
      title_text.compression = PNG_TEXT_COMPRESSION_NONE;
      title_text.key = "Title";
      title_text.text = title;
      png_set_text(png_ptr, info_ptr, &title_text, 1);
   }

   png_write_info(png_ptr, info_ptr);

   // Write image data
   int y;
   for (y = 0 ; y < height ; y++) {
      png_write_row(png_ptr, buffer + y * width * 4);
   }

   // End write
   png_write_end(png_ptr, NULL);

finalise:
   if (fp != NULL) fclose(fp);
   if (info_ptr != NULL) png_free_data(png_ptr, info_ptr, PNG_FREE_ALL, -1);
   if (png_ptr != NULL) png_destroy_write_struct(&png_ptr, (png_infopp)NULL);

   return code;
}
