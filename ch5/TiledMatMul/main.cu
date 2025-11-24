#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cmath>
#include <limits>

#define TILE_WIDTH 16

__global__ void MatrixMulKernel(float* M, float* N, float* P, int m, int n, int o)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	if (row < m && col < o)
	{
		float sum = 0;
		for (int i = 0; i < n; ++i)
		{
			sum += M[row * n + i] * N[i * o + col];
		}
		P[row * o + col] = sum;
	}
}

__global__ void TiledMatrixKernel(float* M, float* N, float* P, int m, int n, int o)
{
	__shared__ float mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ float nds[TILE_WIDTH][TILE_WIDTH];

	int by = blockIdx.y;
	int bx = blockIdx.x;
	int ty = threadIdx.y;
	int tx = threadIdx.x;

	int row = by * TILE_WIDTH + ty;
	int col = bx * TILE_WIDTH + tx;

	float pValue = 0.0f;
	for (int ph = 0; ph < (n + TILE_WIDTH - 1) / TILE_WIDTH; ++ph)
	{
		if (row < m && ph * TILE_WIDTH + tx < n)
		{
			mds[ty][tx] = M[row * n + ph * TILE_WIDTH + tx];
		}
		else
		{
			mds[ty][tx] = 0.0f;
		}
		if (col < o && ph * TILE_WIDTH + ty < n)
		{
			nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * o + col];
		}
		else
		{
			nds[ty][tx] = 0.0f;
		}
		__syncthreads();
		for (int k = 0; k < TILE_WIDTH; ++k)
		{
			pValue += mds[ty][k] * nds[k][tx];
		}
		__syncthreads();
	}

	if (row < m && col < o)
	{
		P[row * o + col] = pValue;
	}
}

void matrixMulGPU(float* M, float* N, float* P, int m, int n, int o)
{
	float* d_M;
	float* d_N;
	float* d_P;

	cudaMalloc((void**)&d_M, m * n * sizeof(float));
	cudaMalloc((void**)&d_N, n * o * sizeof(float));
	cudaMalloc((void**)&d_P, m * o * sizeof(float));

	cudaMemcpy(d_M, M, m * n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_N, N, n * o * sizeof(float), cudaMemcpyHostToDevice);

	dim3 blockDim(16, 16);
	dim3 gridDim((o + blockDim.x - 1) / blockDim.x, (m + blockDim.y - 1) / blockDim.y);

	MatrixMulKernel << <gridDim, blockDim >> > (d_M, d_N, d_P, m, n, o);
	cudaMemcpy(P, d_P, m * o * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_M);
	cudaFree(d_N);
	cudaFree(d_P);
}

void matrixMulTilingGPU(float* M, float* N, float* P, int m, int n, int o)
{
	float* d_M;
	float* d_N;
	float* d_P;

	cudaMalloc((void**)&d_M, m * n * sizeof(float));
	cudaMalloc((void**)&d_N, n * o * sizeof(float));
	cudaMalloc((void**)&d_P, m * o * sizeof(float));

	cudaMemcpy(d_M, M, m * n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_N, N, n * o * sizeof(float), cudaMemcpyHostToDevice);

	dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
	dim3 gridDim((o + blockDim.x - 1) / blockDim.x, (m + blockDim.y - 1) / blockDim.y);
	TiledMatrixKernel << <gridDim, blockDim >> > (d_M, d_N, d_P, m, n, o);
	cudaMemcpy(P, d_P, m * o * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_M);
	cudaFree(d_N);
	cudaFree(d_P);
}

void matrixMulCPU(float* M, float* N, float* P, int m, int n, int o)
{
	for (int i = 0; i < m; ++i)
	{
		for (int j = 0; j < o; ++j)
		{
			float sum = 0.0f;
			for (int k = 0; k < n; ++k)
			{
				sum += M[i * n + k] * N[k * o + j];
			}
			P[i * o + j] = sum;
		}
	}
}

int main()
{
	int m = 2100;
	int n = 3200;
	int o = 9111;

	float* M = new float[m * n];
	float* N = new float[n * o];
	float* P1 = new float[m * o];
	float* P2 = new float[m * o];

	srand(time(NULL));
	for (int i = 0; i < m * n; ++i)
	{
		M[i] = rand() / static_cast<float>(RAND_MAX);
	}
	for (int i = 0; i < n * o; ++i)
	{
		N[i] = rand() / static_cast<float>(RAND_MAX);
	}

	matrixMulGPU(M, N, P1, m, n, o);
	matrixMulTilingGPU(M, N, P2, m, n, o);

	bool check = true;
	float epsilon = 1e-5f;
	for (int i = 0; i < m * o; ++i)
	{
		float diff = std::fabs(P1[i] - P2[i]);

		if (std::fabs(P1[i] - P2[i]) > epsilon)
		{
			check = false;
			break;
		}
	}

	if (!check)
	{
		std::cout << "Failed to compute correctly" << std::endl;
	}
	else
	{
		std::cout << "Correct" << std::endl;
	}

	delete[] M;
	delete[] N;
	delete[] P1;
	delete[] P2;
}
