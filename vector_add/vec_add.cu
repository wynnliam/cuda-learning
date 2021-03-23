// Liam Wynn, 3/23/2021, CUDA Learning

/*
 * Demo taken from Kirk & Hwu's Programming Massively Parallel Processors, Third Edition.
 *
 * To compile do:
 * nvcc vec_add.cu
 *
 * You may get an error about a lack of Microsoft Visual Studio or whatever. In that case
 * do:
 *
 * nvcc -allow-unsupported-compiler vec_add.cu
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

// Add the elements of h_A, h_B and put the result in h_C
void vecAdd(float *h_A, float *h_B, float *h_C, const int n);

// Prints a given vector of size n.
void vecPrint(float *v, const int n);

int main() {
	const int N = 1000000;
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

	printf("h_A: "); vecPrint(h_A, 10);
	printf("h_B: "); vecPrint(h_B, 10);
	printf("h_C: "); vecPrint(h_C, 10);
}

__global__
void vecAddKernel(float *A, float *B, float *C, const int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i < n) C[i] = A[i] + B[i];
}

void vecAdd(float *h_A, float *h_B, float *h_C, const int n) {
  float *d_A, *d_B, *d_C;
  const int size = sizeof(float) * n;
  cudaError_t err;

  err = cudaMalloc((void**)&d_A, size);
  if(err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  } 

  err = cudaMalloc((void**)&d_B, size);
	if(err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	err = cudaMalloc((void**)&d_C, size);
	if(err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

  // Must copy the data from the host to the device
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  // Invoke the kernel. We want ceil(n / 256) blocks, each with 256 threads in them.
  vecAddKernel<<<(int)ceil(n/256.0), 256>>>(d_A, d_B, d_C, n);

  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

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
