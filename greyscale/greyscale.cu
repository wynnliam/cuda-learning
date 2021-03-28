// Liam Wynn, 3/27/2021, CUDA Learning

#include <stdio.h>
#include <stdlib.h>
#define LOADBMP_IMPLEMENTATION
#include "./loadbmp.h"

typedef unsigned int img_dim;

__global__
void greyscaleKernel(unsigned char *pin, unsigned char *pout, const img_dim width, const img_dim height) {
  unsigned int row, col;
  col = blockIdx.x * blockDim.x + threadIdx.x;
  row = blockIdx.y * blockDim.y + threadIdx.y;

  // Index into both arrays.
  unsigned int pixelIndex;
  unsigned char r, g, b;
  unsigned char grey;

  if(col < width && row < height) {
    pixelIndex = (row * width + col) * 3;
    r = pin[pixelIndex];
    g = pin[pixelIndex + 1];
    b = pin[pixelIndex + 2];
    grey = (unsigned char)(0.21f * r + 0.71f * g + 0.07f * b);
    pout[pixelIndex] = grey;
    pout[pixelIndex + 1] = grey;
    pout[pixelIndex + 2] = grey;
  }
}

void greyscale(unsigned char *pin, unsigned char *pout, const img_dim width, const img_dim height) {
  unsigned char *d_pin, *d_pout;
  const int size = sizeof(unsigned char) * width * height * 3;
  dim3 dim_grid(ceil(width / 16.0f), ceil(height / 16.0f), 1);
  dim3 dim_block(16, 16, 1);

  cudaMalloc((void**)&d_pin, size);
  cudaMalloc((void**)&d_pout, size);

  cudaMemcpy(d_pin, pin, size, cudaMemcpyHostToDevice);
  greyscaleKernel<<<dim_grid, dim_block>>>(d_pin, d_pout, width, height);
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

  printf("Image dimensions: %u by %u\n", width, height);

  unsigned char *grey = (unsigned char*)malloc(sizeof(unsigned char) * width * height * 3);
  greyscale(pixels, grey, width, height);
  load_err = loadbmp_encode_file("./grey.bmp", grey, width, height, LOADBMP_RGB);
  if(load_err) {
    printf("LoadBMP Write Error: %u\n", load_err);
  }

  free(pixels);
  free(grey);
  return 0;
}
