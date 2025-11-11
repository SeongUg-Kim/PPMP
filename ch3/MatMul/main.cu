
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define THREADS_PER_BLOCK 256
#define M_ROW 1024
#define WIDTH 2048
#define N_COL 512

#define ERROR_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (code %d) at %s:%d\n", cudaGetErrorString(err), err, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)


__global__ void matrix_mul_kernel(float* m, float* n, float* p, int m_rows, int n_cols, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m_rows && col < n_cols) {
        float p_value = 0.0f;
        for (int t = 0; t < k; ++t) {
            p_value += m[row * k + t] * n[t * n_cols + col];
        }
        p[row * n_cols + col] = p_value;
    }
}

int main(int argc, char* argv[])
{
    int matrix_m_row = M_ROW;
    int matrix_m_col = WIDTH;
    int matrix_n_row = WIDTH;
    int matrix_n_col = N_COL;
    int matrix_p_row = matrix_m_row;
    int matrix_p_col = matrix_n_col;
    int matrix_m_size = matrix_m_row * matrix_m_col;
    int matrix_n_size = matrix_n_row * matrix_n_col;
    int matrix_p_size = matrix_p_row * matrix_p_col;
    float* h_matrix_m = (float*)malloc(sizeof(float) * matrix_m_size);
    float* h_matrix_n = (float*)malloc(sizeof(float) * matrix_n_size);
    float* h_matrix_p = (float*)malloc(sizeof(float) * matrix_p_size);
    float* d_matrix_m;
    float* d_matrix_n;
    float* d_matrix_p;

    ERROR_CHECK(cudaMalloc((void**)&d_matrix_m, sizeof(float) * matrix_m_size));
    ERROR_CHECK(cudaMalloc((void**)&d_matrix_n, sizeof(float) * matrix_n_size));
    ERROR_CHECK(cudaMalloc((void**)&d_matrix_p, sizeof(float) * matrix_p_size));

    srand(time(NULL));
    for (int i = 0; i < matrix_m_row; ++i) {
        for (int j = 0; j < matrix_m_col; ++j) {
            h_matrix_m[i * matrix_m_col + j] = (float)rand();
        }
    }
    for (int i = 0; i < matrix_n_row; ++i) {
        for (int j = 0; j < matrix_n_col; ++j) {
            h_matrix_n[i * matrix_n_col + j] = (float)rand();
        }
    }

    ERROR_CHECK(cudaMemcpy(d_matrix_m, h_matrix_m, matrix_m_size * sizeof(float), cudaMemcpyHostToDevice));
    ERROR_CHECK(cudaMemcpy(d_matrix_n, h_matrix_n, matrix_n_size * sizeof(float), cudaMemcpyHostToDevice));

    dim3 blockDim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 gridDim(ceil((float)matrix_p_col / THREADS_PER_BLOCK), ceil((float)matrix_p_row / THREADS_PER_BLOCK), 1);
    matrix_mul_kernel << <gridDim, blockDim >> > (d_matrix_m, d_matrix_n, d_matrix_p, matrix_m_row, matrix_n_col, matrix_m_col);
    ERROR_CHECK(cudaMemcpy(h_matrix_p, d_matrix_p, matrix_p_size * sizeof(float), cudaMemcpyDeviceToHost));

    free(h_matrix_m);
    free(h_matrix_n);
    free(h_matrix_p);
    cudaFree(d_matrix_m);
    cudaFree(d_matrix_n);
    cudaFree(d_matrix_p);
}