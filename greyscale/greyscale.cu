// Liam Wynn, 3/27/2021, CUDA Learning

#include <stdio.h>
#include <stdlib.h>
#define LOADBMP_IMPLEMENTATION
#include "./loadbmp.h"

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

  free(pixels);
  return 0;
}
