
#include <cinttypes>

#include <cuda.h>

#include <png.h>

__global__ void clover_kernel(uchar4 *image, size_t stride, const dim2 center);
extern int write_png(char* filename, int width, int height, unsigned char *buffer, char* title);

int main(const int argc, const char *argv[])
{
    unsigned char *output;
    unsigned char *device_output;

    const size_t height = 1024;
    const size_t width  = 1024;
    const size_t size   = 4 * height * width;
    output = new uchar[size];

    CUDA_ASSERT(cudaMalloc((void **)&device_output, size * sizeof(uchar)), "");

    CUDA_ASSERT(cudaMemcpy(device_output, output, size * sizeof(uchar), cudaMemcpyHostToDevice), "");

    dim3 blocks(1, 1, 1);

    CUDA_ASSERT(cudaMemcpy(output, device_output, size * sizeof(uchar), cudaMemcpyDeviceToHost), "");
    write_png("clover.png", width, height, output, NULL);

    CUDA_ASSERT(cudaFree(device_output), "");

    delete[] output;

    return 0;
}

int write_png(char* filename, int width, int height, unsigned char *buffer, char* title)
{
   int code = 0;
   FILE *fp = NULL;
   png_structp png_ptr = NULL;
   png_infop info_ptr = NULL;
   png_bytep row = NULL;

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
         8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
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
   int x, y;
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