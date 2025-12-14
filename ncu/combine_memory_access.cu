#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>

//合并访存
__global__ void add1(float* x, float* y, float* z){
    //计算一维全局idx索引
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    z[n] = x[n] + y[n];
}
//连续未对齐(只使用L2,32对齐；使用了L1、L2,128对齐)
__global__ void add2(float* x, float* y, float* z){
    //计算一维全局idx索引,访存地址偏移1，不再对齐
    int n = blockIdx.x * blockDim.x + threadIdx.x + 1;
    z[n] = x[n] + y[n];
}

//交错不连续访存,也可以合并访存
__global__ void add3(float* x, float* y, float* z){
    //计算一维全局idx索引,访存地址偏移1，不再对齐
    int tid_permute = threadIdx.x ^ 0x1;
    int n = blockIdx.x * blockDim.x + tid_permute;
    z[n] = x[n] + y[n];
}

//所有线程都访问一个数，访存利用率低
__global__ void add4(float* x, float* y, float* z){
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_idx = n / 32;
    z[warp_idx] = x[warp_idx] + y[warp_idx];
}

//不连续访存
__global__ void add5(float* x, float* y, float* z){
    //计算一维全局idx索引,访存地址偏移1，不再对齐
    int n = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    z[n] = x[n] + y[n];
}

int main(){
    const int N = 32 * 1024 * 1024;
    float *input_x = (float*)malloc(N * sizeof(float));
    float *input_y = (float*)malloc(N * sizeof(float));
    float *output = (float*)malloc(N * sizeof(float));

    float *d_input_x, *d_input_y, *d_output;

    cudaMalloc((void **)&d_input_x, N * sizeof(float));
    cudaMalloc((void **)&d_input_y, N * sizeof(float));
    cudaMalloc((void **)&d_output, N * sizeof(float));

    cudaMemcpy(d_input_x, input_x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_y, input_y, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid(N / 256);
    dim3 block(64); //每个block只计算四分之一

    for(int i = 0; i < 2; i++){
        add1<<<grid, block>>>(d_input_x, d_input_y, d_output);
    }

    for(int i = 0; i < 2; i++){
        add2<<<grid, block>>>(d_input_x, d_input_y, d_output);
    }

    for(int i = 0; i < 2; i++){
        add3<<<grid, block>>>(d_input_x, d_input_y, d_output);
    }

    for(int i = 0; i < 2; i++){
        add4<<<grid, block>>>(d_input_x, d_input_y, d_output);
    }

    for(int i = 0; i < 2; i++){
        add5<<<grid, block>>>(d_input_x, d_input_y, d_output);
    }
}