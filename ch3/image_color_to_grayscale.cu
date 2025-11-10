#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define CHANNELS 3

__global__ void color_to_grayscale_conversion(
    unsigned char* out, 
    unsigned char* in,
    int width,
    int height)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < width && row < height) {
        // get 1D offset for the grayscale image
        int gray_offset = row * width + col;
        int rgb_offset = gray_offset * CHANNELS;
        unsigned char r = in[rgb_offset];
        unsigned char g = in[rgb_offset + 1];
        unsigned char b = in[rgb_offset + 2];
        out[gray_offset] = 0.21f * r * 0.72f * g + 0.007 * b;
    }
}

int main(int argc, char* argv[])
{
    int block_dim;
    int image_width;
    int image_height;
    unsigned char* h_input_image;
    unsigned char* h_output_image;
    unsigned char* d_input_image;
    unsigned char* d_output_image;
    int size;

    if (argc != 4) {
        printf("Usage: %s <block_dim> <image_width> <image_height>\n", argv[0]);
        return 1;
    }

    block_dim = atoi(argv[1]);
    image_width = atoi(argv[2]);
    image_height = atoi(argv[3]);
    size = image_height * image_width;

    // allocate memory for the input and output images on host
    h_input_image = (unsigned char*)malloc(sizeof(unsigned char) * size * CHANNELS);
    h_output_image = (unsigned char*)malloc(sizeof(unsigned char) * size);

    // allocate memory for the input and output images on device
    cudaMalloc((void**)&d_input_image, CHANNELS * size * sizeof(unsigned char));
    cudaMalloc((void**)&d_output_image, size * sizeof(unsigned char));

    // init the input image with rand value between 0 and 255
    srand(time(NULL));
    for (int i = 0; i < CHANNELS * size; ++i) {
        h_input_image[i] = rand() % 256;
    }
    cudaMemcpy(d_input_image, h_input_image, sizeof(unsigned char) * size * CHANNELS, cudaMemcpyHostToDevice);

    dim3 dimBlock(block_dim, block_dim, 1);
    dim3 dimGrid(ceil((float)image_width / block_dim), ceil((float)image_height / block_dim), 1);
    color_to_grayscale_conversion<<<dimGrid, dimBlock>>>(d_input_image, d_output_image, image_width,image_height);

    cudaMemcpy(h_output_image, d_output_image, sizeof(unsigned char) * size, cudaMemcpyDeviceToHost);

    free(h_input_image);
    free(h_output_image);
    cudaFree(d_input_image);
    cudaFree(d_output_image);
}