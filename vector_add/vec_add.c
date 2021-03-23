// Liam Wynn, 3/23/2021, CUDA Learning

/*
 * Demo taken from Kirk & Hwu's Programming Massively Parallel Processors, Third Edition.
 *
 * To compile do:
 * nvcc vec_add.c
 *
 * You may get an error about a lack of Microsoft Visual Studio or whatever. In that case
 * do:
 *
 * nvcc -allow-unsupported-compiler vec_add.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

// Add the elements of h_A, h_B and put the result in h_C
void vecAdd(float *h_A, float *h_B, float *h_C, const int n);

// Prints a given vector of size n.
void vecPrint(float *v, const int n);

int main() {
	const int N = 10;
	float *h_A, *h_B, *h_C;

	h_A = (float*)malloc(N * sizeof(float));
	h_B = (float*)malloc(N * sizeof(float));
	h_C = (float*)malloc(N * sizeof(float));
	int i;
  for(i = 0; i < N; i++) {
    h_A[i] = i;
    h_B[i] = 1;
    h_C[i] = 0;
  }

	vecAdd(h_A, h_B, h_C, N);

	printf("h_A: "); vecPrint(h_A, N);
	printf("h_B: "); vecPrint(h_B, N);
	printf("h_C: "); vecPrint(h_C, N);
}

void vecAdd(float *h_A, float *h_B, float *h_C, const int n) {
  float *d_A, *d_B, *d_C;

  cudaMalloc((void**)&d_A, sizeof(float) * n);
  cudaMalloc((void**)&d_B, sizeof(float) * n);
  cudaMalloc((void**)&d_C, sizeof(float) * n);

  int i;
  for(i = 0; i < n; i++)
    h_C[i] = h_A[i] + h_B[i];

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

void vecPrint(float *v, const int n) {
	int i;
	for(i = 0; i < n; i++)
		printf("%f,", v[i]);
	printf("\n");
}
