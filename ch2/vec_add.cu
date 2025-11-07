#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define THREADS_PER_BLOCK 256
#define NUM_THREADS (1024 * 1024)

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (code %d) at %s:%d\n", cudaGetErrorString(err), err, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

__global__ void vecAddKernel (float* a, float* b, float* c, int n) 
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx]  + b[idx];
    }
}

void vecAdd(float* a, float* b, float* c, int n)
{
    int size = n * sizeof(float);
    float* dA;
    float* dB;
    float* dC;

    // allocate device memory 
    CUDA_CHECK(cudaMalloc((void**)&dA, size));
    CUDA_CHECK(cudaMalloc((void**)&dB, size));
    CUDA_CHECK(cudaMalloc((void**)&dC, size));

    // copy inputs to device
    CUDA_CHECK(cudaMemcpy(dA, a, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, b, size, cudaMemcpyHostToDevice));

    // kernel function call
    dim3 gridDim = ceil((float)n / THREADS_PER_BLOCK);
    dim3 blockDim = THREADS_PER_BLOCK;
    vecAddKernel<<<gridDim, blockDim>>>(dA, dB, dC, n);

    // copy result to host
    CUDA_CHECK(cudaMemcpy(c, dC, size, cudaMemcpyDeviceToHost));

    // free device memory
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
}

int main(void)
{
    float* a;
    float* b;
    float* c;
    float* hc;
    int n = NUM_THREADS;
    float epsilon = 1e-6;
    a = (float*)malloc(n * sizeof(float));
    b = (float*)malloc(n * sizeof(float));
    c = (float*)malloc(n * sizeof(float));
    hc = (float*) malloc(n * sizeof(float));
    for (int i = 0; i < n; ++i) {
        a[i] = (float)rand() / (float)RAND_MAX;
        b[i] = (float)rand() / (float)RAND_MAX;
    }
    for (int i = 0; i < n; ++i) {
        hc[i] = a[i] + b[i];
    }
    printf("start vec_add\n");
    vecAdd(a, b, c, n); 
    printf("end vec_add\n");
    int res = 1;
    for (int i = 0; i < n; ++i) {
        if (fabs(c[i] - hc[i]) > epsilon) {
            res = 0;
            break;
        }
    }
    if (res) {
        printf("Kernel function works well");
    } else {
        printf("Kernel function does not work correctly");
    }
    free(a);
    free(b);
    free(c);
    free(hc);
}