// Liam Wynn, 3/30/2021, CUDA Learning

/*
 * Demo taken from Kirk & Hwu's Programming Massively Parallel Processors, Third Edition.
 *
 * To compile do:
 * nvcc blur.cu
 *
 * You may get an error about a lack of Microsoft Visual Studio or whatever. In that case
 * do:
 *
 * nvcc -allow-unsupported-compiler blur.cu
 */

#include <stdio.h>
#include <stdlib.h>
#define LOADBMP_IMPLEMENTATION
#include "./loadbmp.h"

// Number of pixels around the main pixel
// we compute the average of.
#define BLUR_SIZE	1

typedef int img_dim;

__global__
void blurKernel(unsigned char *pin, unsigned char *pout, const img_dim width, const img_dim height) {
  int row, col;
  col = blockIdx.x * blockDim.x + threadIdx.x;
  row = blockIdx.y * blockDim.y + threadIdx.y;

  unsigned int r_val, g_val, b_val;
  unsigned int pixel_index, curr_pixel_index;
  unsigned int pixel_count;

  int blur_row, blur_col;
  int curr_row, curr_col;

  if(col < width && row < height) {
    // Each column is 3 pixels.
    pixel_index = row * (width * 3) + (col * 3);
    r_val = 0;
    g_val = 0;
    b_val = 0;

    pixel_count = 0;

    for(blur_row = -BLUR_SIZE; blur_row < BLUR_SIZE + 1; blur_row++) {
      for(blur_col = -BLUR_SIZE; blur_col < BLUR_SIZE + 1; blur_col++) {
        curr_row = row + blur_row;
        curr_col = (col + blur_col) * 3;

        if(curr_row > -1 && curr_row < height && curr_col > -1 && curr_col < (width * 3)) {
          curr_pixel_index = curr_row * (width * 3) + curr_col;
          r_val += pin[curr_pixel_index + 0];
          g_val += pin[curr_pixel_index + 1];
          b_val += pin[curr_pixel_index + 2];
          pixel_count++;
        }
      }
    }

    pout[pixel_index + 0] = (unsigned char)(r_val / pixel_count);
    pout[pixel_index + 1] = (unsigned char)(g_val / pixel_count);
    pout[pixel_index + 2] = (unsigned char)(b_val / pixel_count);
  }
}

void blur (unsigned char *pin, unsigned char *pout, const img_dim width, const img_dim height) {
  unsigned char *d_pin, *d_pout;
  const int size = sizeof(unsigned char) * width * height * 3;
  dim3 dim_grid(ceil(width / 16.0f), ceil(height / 16.0f), 1);
  dim3 dim_block(16, 16, 1);

  cudaMalloc((void**)&d_pin, size);
  cudaMalloc((void**)&d_pout, size);

  cudaMemcpy(d_pin, pin, size, cudaMemcpyHostToDevice);
  blurKernel<<<dim_grid, dim_block>>>(d_pin, d_pout, width, height);
  cudaMemcpy(pout, d_pout, size, cudaMemcpyDeviceToHost);

  cudaFree(d_pin);
  cudaFree(d_pout);
}

int main() {
  const char *image_path = "./sample.bmp";
  unsigned char *pixels = NULL;
  unsigned int width, height;

  unsigned int load_err = loadbmp_decode_file(image_path, &pixels, &width, &height, LOADBMP_RGB);
  if(load_err) {
    printf("LoadBMP Load Error: %u\n", load_err);
    exit(-1);
  }

  unsigned char *blurred = (unsigned char*)malloc(sizeof(unsigned char) * width * height * 3);
  blur(pixels, blurred , (int)width, (int)height);
  load_err = loadbmp_encode_file("./blur.bmp", blurred, width, height, LOADBMP_RGB);
  if(load_err) {
    printf("LoadBMP Write Error: %u\n", load_err);
  }

  free(pixels);
  free(blurred);
  return 0;
}

