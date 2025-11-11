#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define BLUR_SIZE 1

__global__ void blur_kernel(unsigned char* in, unsigned char* out, int w, int h)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < w && row < h) {
        int pix_val = 0;
        int pixels = 0;
        for (int blur_row = -BLUR_SIZE; blur_row < BLUR_SIZE + 1; ++blur_row) {
            for (int blur_col = -BLUR_SIZE; blur_col < BLUR_SIZE + 1; ++blur_col) {
                int cur_row = row + blur_row;
                int cur_col = col + blur_col;
                if (cur_row < 0 || cur_row >= h || cur_col < 0 || cur_col >= w) {
                    continue;
                }
                pix_val += in[cur_row * w + cur_col];
                ++pixels;
            }
        }
        out[row * w + col] = (unsigned char)(pix_val / pixels);
    }
}

int main(int argc, char* argv[])
{
    int block_dim;
    int image_width;
    int image_height;
    int image_size;

    unsigned char* h_image_input;
    unsigned char* h_image_output;
    unsigned char* d_image_input;
    unsigned char* d_image_output;

    if (argc != 4) {
        printf("Usage: %s <block_dim> <image_width> <image_height>", argv[0]);
        return 1;
    }

    block_dim = atoi(argv[1]);
    image_width = atoi(argv[2]);
    image_height = atoi(argv[3]);
    image_size = image_width * image_height;

    h_image_input = (unsigned char*)malloc(image_size * sizeof(unsigned char));
    h_image_output = (unsigned char*)malloc(image_size * sizeof(unsigned char));
    cudaMalloc((void**)&d_image_input, image_size * sizeof(unsigned char));
    cudaMalloc((void**)&d_image_output, image_size * sizeof(unsigned char));
    
    srand(time(NULL));
    for (int i = 0; i < image_size; ++i) {
        h_image_input[i] = rand() % 256;
    }

    cudaMemcpy(d_image_input, h_image_input, sizeof(unsigned char) * image_size, cudaMemcpyHostToDevice);
    dim3 gridDim(ceil((float)image_width / block_dim), ceil((float)image_height / block_dim), 1);
    dim3 blockDim(block_dim, block_dim, 1);
    blur_kernel<<<gridDim, blockDim>>>(d_image_input, d_image_output, image_width, image_height);
    cudaMemcpy(h_image_output, d_image_output, sizeof(unsigned char) * image_size, cudaMemcpyDeviceToHost);

    free(h_image_input);
    free(h_image_output);
    cudaFree(d_image_input);
    cudaFree(d_image_output);
}