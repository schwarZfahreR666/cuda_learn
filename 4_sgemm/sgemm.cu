#include <cuda_runtime.h>
#include "../include/utils.h"

void random_matrix(int m, int n, float* matrix){
    #define A(i, j) matrix[(i) * (n) + (j)]
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            A(i, j) = 2.0 * (float)drand48() - 1.0;
        }
    }
}

float compare_matrices(int m, int n, float* a, float* b){
    float max_dif = 0.0, diff;
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            // printf("%f %f\n", a[i * n + j], b[i * n + j]);
            diff = abs(a[i * n + j] - b[i * n + j]);
            max_dif = (diff > max_dif ? diff : max_dif);
        }
    }
    return max_dif;
}

void cpu_sgemm(float* A, float* B, float* C, const int M, const int N, const int K){
    for(int m = 0; m < M; m++){
        for(int n = 0; n < N; n++){
            float temp = 0.f;
            for(int k = 0; k < K; k++){
                temp += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = temp;
        }
    }
}

//假设有m*n个线程，每个线程计算一个C中的元素
__global__ void simple_sgemm(float* A, float* B, float* C, const int M, const int N, const int K){
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    float* A_ptr_start = A + blockDim.y * blockIdx.y * K;
    float* B_ptr_start = B + blockDim.x * blockIdx.x;
    float temp = 0.f;
    for(int k = 0; k < K; k++){
        int a_idx = threadIdx.y * K + k;
        int b_idx = threadIdx.x + k * N;
        temp += A_ptr_start[a_idx] * B_ptr_start[b_idx];
        
    }
    C[y * N + x] = temp;
}

int main(){
    int m = 16;
    int n = 16;
    int k = 16;
    const size_t mem_size_A = m * k * sizeof(float);
    const size_t mem_size_B = k * n * sizeof(float);
    const size_t mem_size_C = m * n * sizeof(float);

    float* matrix_A_host = (float*)malloc(mem_size_A);
    float* matrix_B_host = (float*)malloc(mem_size_B);
    random_matrix(m, n, matrix_A_host);
    random_matrix(n, k, matrix_B_host);

    float* matrix_C_host_cpu = (float*)malloc(mem_size_C);
    float* matrix_C_host_gpu = (float*)malloc(mem_size_C);
    memset(matrix_C_host_cpu, 0, mem_size_C);
    memset(matrix_C_host_gpu, 0, mem_size_C);

    float *matrix_A_device = 0, *matrix_B_device = 0, *matrix_C_device = 0;
    cudaMalloc((void **)&matrix_A_device, mem_size_A);
    cudaMalloc((void **)&matrix_B_device, mem_size_B);
    cudaMalloc((void **)&matrix_C_device, mem_size_C);

    cudaMemcpy(matrix_A_device, matrix_A_host, mem_size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(matrix_B_device, matrix_B_host, mem_size_B, cudaMemcpyHostToDevice);

    cpu_sgemm(matrix_A_host, matrix_B_host, matrix_C_host_cpu, m, n, k);

    constexpr size_t BLOCK_SIZE = 16;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    //二维网格，分别也和m、n的大小有关
    dim3 grid((m + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    simple_sgemm<<<grid, block>>>(matrix_A_device, matrix_B_device, matrix_C_device, m, n, k);
    cudaDeviceSynchronize();
    cudaMemcpy(matrix_C_host_gpu, matrix_C_device, mem_size_C, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    float max_diff = compare_matrices(m, n, matrix_C_host_cpu, matrix_C_host_gpu);
    printf("max_diff: %f\n", max_diff);

    free(matrix_A_host);
    free(matrix_B_host);
    free(matrix_C_host_cpu);
    free(matrix_C_host_gpu);

    cudaFree(matrix_A_device);
    cudaFree(matrix_B_device);
    cudaFree(matrix_C_device);
}