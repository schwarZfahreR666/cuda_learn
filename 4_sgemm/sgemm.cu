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
            
            diff = abs(a[i * n + j] - b[i * n + j]);
            if(diff > max_dif){
                max_dif = diff;
            }
            if(diff > 1){
                printf("i[%d] j[%d] got %f %f\n",i, j, a[i * n + j], b[i * n + j]);
                return max_dif;
            }
            
            
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
    const int64_t x = threadIdx.x + blockDim.x * blockIdx.x;
    const int64_t y = threadIdx.y + blockDim.y * blockIdx.y;
    float* A_ptr_start = A + blockDim.y * blockIdx.y * K;
    float* B_ptr_start = B + blockDim.x * blockIdx.x;
    float temp = 0.f;
    for(int64_t k = 0; k < K; k++){
        int64_t a_idx = threadIdx.y * K + k;
        int64_t b_idx = threadIdx.x + k * N;
        temp += A_ptr_start[a_idx] * B_ptr_start[b_idx];
        // printf("temp:%f\n", temp);
        
    }
    C[y * N + x] = temp;
}

//使用shared memory，读取内存次数从2KMN降为KMN(1/Bn+1/Bm)
//但是每个block都加载block_size * k的数据，使用了太多共享内存
// template<unsigned int BLOCK_SIZE_X, unsigned int BLOCK_SIZE_Y, unsigned int K_>
// __global__ void shared_mm_sgemm_v1(float* A, float* B, float* C, const int M, const int N, const int K){
//     const int x = threadIdx.x + blockDim.x * blockIdx.x;
//     const int y = threadIdx.y + blockDim.y * blockIdx.y;
//     float* A_ptr_start = A + blockDim.y * blockIdx.y * K;
//     float* B_ptr_start = B + blockDim.x * blockIdx.x;

//     __shared__ float a_shared[BLOCK_SIZE_X][K_];
//     __shared__ float b_shared[K_][BLOCK_SIZE_Y];
    
//     for(int s = 0; s < K; s += blockDim.x){
//         a_shared[threadIdx.y][s + threadIdx.x] = A_ptr_start[threadIdx.y * K + s + threadIdx.x];
//     }
//     for(int s = 0; s < K; s += blockDim.y){
//         b_shared[s + threadIdx.y][threadIdx.x] = B_ptr_start[(s + threadIdx.y) * N + threadIdx.x];
//     }
//     __syncthreads();

//     float temp = 0.f;
//     for(int k = 0; k < K; k++){
//         temp += a_shared[threadIdx.y][k] * b_shared[k][threadIdx.x];
        
//     }
//     C[y * N + x] = temp;
// }
//使用滑动窗口的方式，将矩阵按K维分块计算矩阵乘，减少共享内存的使用
template<unsigned int BLOCK_SIZE>
__global__ void shared_mm_sgemm_v2(float* A, float* B, float* C, const int M, const int N, const int K){
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    float* A_ptr_start = A + blockDim.y * blockIdx.y * K;
    float* B_ptr_start = B + blockDim.x * blockIdx.x;

    __shared__ float a_shared[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float b_shared[BLOCK_SIZE][BLOCK_SIZE];

    float temp = 0.f;
    
    for(int s = 0; s < K; s += BLOCK_SIZE){
        a_shared[threadIdx.y][threadIdx.x] = A_ptr_start[threadIdx.y * K + s + threadIdx.x];
        b_shared[threadIdx.y][threadIdx.x] = B_ptr_start[(threadIdx.y + s) * N + threadIdx.x];
        __syncthreads();
        //计算小矩阵之间的mm
        for(int i = 0; i < BLOCK_SIZE; i++){
            temp += a_shared[threadIdx.y][i] * b_shared[i][threadIdx.x];
        }
        __syncthreads();
    }
    
    C[y * N + x] = temp;
}

//每个block处理多个数据块,该实现每个block计算连续的STRIDE个块
template<unsigned int BLOCK_SIZE, unsigned int STRIDE>
__global__ void shared_mm_sgemm_v3(float* A, float* B, float* C, const int M, const int N, const int K){
    //每个block负责数据的单个维度
    constexpr int STEP = BLOCK_SIZE * STRIDE;
    float* A_ptr_start = A + STEP * blockIdx.y * K;
    float* B_ptr_start = B + STEP * blockIdx.x;

    __shared__ float a_shared[STEP][STEP];
    __shared__ float b_shared[STEP][STEP];

    float temp[STRIDE][STRIDE] = {0.f};
    
    for(int s = 0; s < K; s += STEP){
        for(int i = 0; i < STRIDE; i++){
            for(int j = 0; j < STRIDE; j++){
                a_shared[i * BLOCK_SIZE + threadIdx.y][j * BLOCK_SIZE + threadIdx.x] = A_ptr_start[(threadIdx.y + BLOCK_SIZE * i) * K + s + threadIdx.x + BLOCK_SIZE * j];
                b_shared[i * BLOCK_SIZE + threadIdx.y][j * BLOCK_SIZE + threadIdx.x] = B_ptr_start[(threadIdx.y + BLOCK_SIZE * i + s) * N + threadIdx.x + BLOCK_SIZE * j];
            }
        }
        __syncthreads();
        //计算小矩阵之间的mm
        for(int i = 0; i < STRIDE; i++){
            for(int j = 0; j < STRIDE; j++){
                for(int k = 0; k < STEP; k++){
                    temp[i][j] += a_shared[threadIdx.y + BLOCK_SIZE * i][k] * b_shared[k][threadIdx.x + BLOCK_SIZE * j];
                }
            }
        }
        __syncthreads();
    }
    for(int i = 0; i < STRIDE; i++){
        for(int j = 0; j < STRIDE; j++){
            //每个维度考虑全局block偏移+block内偏移
            int x = threadIdx.x + BLOCK_SIZE * j + STEP * blockIdx.x;
            int y = threadIdx.y + BLOCK_SIZE * i + STEP * blockIdx.y;
            C[y * N + x] = temp[i][j];    
        }
    }
    
}

//每个block处理多个数据块，该实现每个block计算相隔M / STRIDE的块
template<unsigned int BLOCK_SIZE, unsigned int STRIDE>
__global__ void shared_mm_sgemm_v3_skip(float* A, float* B, float* C, const int M, const int N, const int K){
    //每个block负责数据的单个维度
    constexpr int STEP = BLOCK_SIZE * STRIDE;
    float* A_ptr_start = A + BLOCK_SIZE * blockIdx.y * K;
    float* B_ptr_start = B + BLOCK_SIZE * blockIdx.x;

    int BLOCK_STEP = M / STRIDE;

    __shared__ float a_shared[STEP][STEP];
    __shared__ float b_shared[STEP][STEP];

    float temp[STRIDE][STRIDE] = {0.f};
    
    for(int s = 0; s < K; s += BLOCK_SIZE){
        for(int i = 0; i < STRIDE; i++){
            for(int j = 0; j < STRIDE; j++){
                a_shared[i * BLOCK_SIZE + threadIdx.y][j * BLOCK_SIZE + threadIdx.x] = A_ptr_start[(threadIdx.y + BLOCK_STEP * i) * K + s + threadIdx.x];
                b_shared[i * BLOCK_SIZE + threadIdx.y][j * BLOCK_SIZE + threadIdx.x] = B_ptr_start[(threadIdx.y + s) * N + threadIdx.x + BLOCK_STEP * j];
            }
        }
        __syncthreads();
        //计算小矩阵之间的mm
        for(int i = 0; i < STRIDE; i++){
            for(int j = 0; j < STRIDE; j++){
                for(int k = 0; k < BLOCK_SIZE; k++){
                    temp[i][j] += a_shared[threadIdx.y + BLOCK_SIZE * i][k] * b_shared[k][threadIdx.x + BLOCK_SIZE * j];
                }
            }
        }
        __syncthreads();
    }
    for(int i = 0; i < STRIDE; i++){
        for(int j = 0; j < STRIDE; j++){
            //每个维度考虑全局block偏移+block内偏移
            int x = threadIdx.x + BLOCK_STEP * j + BLOCK_SIZE * blockIdx.x;
            int y = threadIdx.y + BLOCK_STEP * i + BLOCK_SIZE * blockIdx.y;
            C[y * N + x] = temp[i][j];    
        }
    }
    
}

//使用float4按行取数
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
template<unsigned int M_NUM_PER_BLOCK, 
        unsigned int N_NUM_PER_BLOCK, 
        unsigned int K_NUM_PER_BLOCK, 
        unsigned int NUM_PER_THREAD>
__global__ void shared_mm_sgemm_float4(float* A, float* B, float* C, const int M, const int N, const int K){
    //每个block负责数据的单个维度
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    float* A_ptr_start = A + M_NUM_PER_BLOCK * blockIdx.y * K;
    float* B_ptr_start = B + N_NUM_PER_BLOCK * blockIdx.x;

    __shared__ float a_shared[M_NUM_PER_BLOCK][K_NUM_PER_BLOCK];
    __shared__ float b_shared[K_NUM_PER_BLOCK][N_NUM_PER_BLOCK];

    float temp[NUM_PER_THREAD] = {0.f};
    
    for(int s = 0; s < K; s += K_NUM_PER_BLOCK){
        //a_shared[ty][tx * NUM_PER_THREAD + (0/1/2/3)] = A_ptr_start[ty * K + s + threadIdx.x * NUM_PER_THREAD + (0/1/2/3)]
        FETCH_FLOAT4(a_shared[ty][tx * NUM_PER_THREAD]) = FETCH_FLOAT4(A_ptr_start[ty * K + s + threadIdx.x * NUM_PER_THREAD]);
        
        FETCH_FLOAT4(b_shared[ty][tx * NUM_PER_THREAD]) = FETCH_FLOAT4(B_ptr_start[(ty + s) * N + tx * NUM_PER_THREAD]);
            
        __syncthreads();
        //计算小矩阵之间的mm
        for(int i = 0; i < NUM_PER_THREAD; i++){
            for(int k = 0; k < K_NUM_PER_BLOCK; k++){
                temp[i] += a_shared[ty][k] * b_shared[k][tx * NUM_PER_THREAD + i];
            }
        }
        __syncthreads();
    }
    float* c_ptr_start = C + blockIdx.y * N * M_NUM_PER_BLOCK + blockIdx.x * N_NUM_PER_BLOCK;
    for(int i = 0; i < NUM_PER_THREAD; i++){
        //每个维度考虑全局block偏移+block内偏移
        c_ptr_start[ty * N + tx * NUM_PER_THREAD + i] = temp[i];    
    }
    
}

template<unsigned int M_NUM_PER_BLOCK, 
        unsigned int N_NUM_PER_BLOCK, 
        unsigned int K_NUM_PER_BLOCK, 
        unsigned int NUM_PER_THREAD>
__global__ void shared_mm_sgemm_float4_register(float* A, float* B, float* C, const int M, const int N, const int K){
    //每个block负责数据的单个维度
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    //计算全局线程索引，然后重排成ctx * cty的方block
    int tid = ty * blockDim.x + tx;
    int ctx = tid % 16;
    int cty = tid / 16;
    float* A_ptr_start = A + M_NUM_PER_BLOCK * blockIdx.y * K;
    float* B_ptr_start = B + N_NUM_PER_BLOCK * blockIdx.x;

    __shared__ float a_shared[M_NUM_PER_BLOCK][K_NUM_PER_BLOCK];
    __shared__ float b_shared[K_NUM_PER_BLOCK][N_NUM_PER_BLOCK];
    
    //每个线程依然负责计算NUM_PER_THREAD个数
    constexpr int REG_NUM = NUM_PER_THREAD / 2;
    float temp[REG_NUM][REG_NUM] = {0.f};
    float a_reg[REG_NUM] = {0.f};
    float b_reg[REG_NUM] = {0.f};
    
    for(int s = 0; s < K; s += K_NUM_PER_BLOCK){
        //a_shared[ty][tx * NUM_PER_THREAD + (0/1/2/3)] = A_ptr_start[ty * K + s + threadIdx.x * NUM_PER_THREAD + (0/1/2/3)]
        FETCH_FLOAT4(a_shared[ty][tx * NUM_PER_THREAD]) = FETCH_FLOAT4(A_ptr_start[ty * K + s + threadIdx.x * NUM_PER_THREAD]);
        
        FETCH_FLOAT4(b_shared[ty][tx * NUM_PER_THREAD]) = FETCH_FLOAT4(B_ptr_start[(ty + s) * N + tx * NUM_PER_THREAD]);
            
        __syncthreads();
        //计算小矩阵之间的mm
        for(int k = 0; k < K_NUM_PER_BLOCK; k++){
            //先把固定k的位置的数加载到寄存器
            //一共32*32个数，16*16个线程。
            //每个线程负责计算四个数，x、y方向各加载两个数
            a_reg[0] = a_shared[cty * 2][k];
            a_reg[1] = a_shared[cty * 2 + 1][k];
            b_reg[0] = b_shared[k][ctx * 2];
            b_reg[1] = b_shared[k][ctx * 2 + 1];
            for(int i = 0; i < REG_NUM; i++){
                for(int j = 0; j < REG_NUM; j++){
                    temp[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }
        
        __syncthreads();
    }
    float* c_ptr_start = C + blockIdx.y * N * M_NUM_PER_BLOCK + blockIdx.x * N_NUM_PER_BLOCK;
    for(int i = 0; i < REG_NUM; i++){
        for(int j = 0; j < REG_NUM; j++){
            //每个维度考虑全局block偏移+block内偏移
            //每个thread将负责计算的4个数写回全局内存
            //block偏移已经算好了，计算block内偏移即可
            c_ptr_start[(cty * 2 + i) * N + ctx * 2 + j] = temp[i][j];
        }    
    }
    
}

int main(){
    double iStart, iElaps;

    int m = 1024;
    int n = 1024;
    constexpr int k = 1024;
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
    iStart = cpuSecond();
    cpu_sgemm(matrix_A_host, matrix_B_host, matrix_C_host_cpu, m, n, k);
    iElaps = cpuSecond() - iStart;
    printf("cpu_sgemm time: %lf\n", iElaps);
    constexpr size_t BLOCK_SIZE = 32;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    //二维网格，分别也和m、n的大小有关
    dim3 grid((m + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    cudaMemset(matrix_C_device, 0, mem_size_C);
    memset(matrix_C_host_gpu, 0, mem_size_C);
    iStart = cpuSecond();
    simple_sgemm<<<grid, block>>>(matrix_A_device, matrix_B_device, matrix_C_device, m, n, k);
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     printf("Launch failed: %s\n", cudaGetErrorString(err));
    // }
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    cudaMemcpy(matrix_C_host_gpu, matrix_C_device, mem_size_C, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    float max_diff = compare_matrices(m, n, matrix_C_host_cpu, matrix_C_host_gpu);
    printf("simple_sgemm time: %lf , max_diff: %f\n", iElaps, max_diff);

    // cudaMemset(matrix_C_device, 0, mem_size_C);
    // shared_mm_sgemm_v1<BLOCK_SIZE, BLOCK_SIZE, k><<<grid, block>>>(matrix_A_device, matrix_B_device, matrix_C_device, m, n, k);
    // cudaDeviceSynchronize();
    // cudaMemcpy(matrix_C_host_gpu, matrix_C_device, mem_size_C, cudaMemcpyDeviceToHost);
    // cudaDeviceSynchronize();
    // max_diff = compare_matrices(m, n, matrix_C_host_cpu, matrix_C_host_gpu);
    // printf("max_diff: %f\n", max_diff);

    cudaMemset(matrix_C_device, 0, mem_size_C);
    memset(matrix_C_host_gpu, 0, mem_size_C);
    iStart = cpuSecond();
    shared_mm_sgemm_v2<BLOCK_SIZE><<<grid, block>>>(matrix_A_device, matrix_B_device, matrix_C_device, m, n, k);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    cudaMemcpy(matrix_C_host_gpu, matrix_C_device, mem_size_C, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    max_diff = compare_matrices(m, n, matrix_C_host_cpu, matrix_C_host_gpu);
    printf("shared_mm_sgemm_v2 time: %lf , max_diff: %f\n", iElaps, max_diff);

    dim3 block_v3(BLOCK_SIZE, BLOCK_SIZE);
    constexpr int STRIDE = 2;
    //二维网格，分别也和m、n的大小有关
    //只使用原来的四分之一block
    dim3 grid_v3((m + BLOCK_SIZE - 1) / BLOCK_SIZE / STRIDE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE / STRIDE);
    cudaMemset(matrix_C_device, 0, mem_size_C);
    memset(matrix_C_host_gpu, 0, mem_size_C);
    iStart = cpuSecond();
    shared_mm_sgemm_v3<BLOCK_SIZE, STRIDE><<<grid_v3, block_v3>>>(matrix_A_device, matrix_B_device, matrix_C_device, m, n, k);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    cudaMemcpy(matrix_C_host_gpu, matrix_C_device, mem_size_C, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    max_diff = compare_matrices(m, n, matrix_C_host_cpu, matrix_C_host_gpu);
    printf("shared_mm_sgemm_v3 time: %lf , max_diff: %f\n", iElaps, max_diff);

    cudaMemset(matrix_C_device, 0, mem_size_C);
    memset(matrix_C_host_gpu, 0, mem_size_C);
    iStart = cpuSecond();
    shared_mm_sgemm_v3_skip<BLOCK_SIZE, STRIDE><<<grid_v3, block_v3>>>(matrix_A_device, matrix_B_device, matrix_C_device, m, n, k);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    cudaMemcpy(matrix_C_host_gpu, matrix_C_device, mem_size_C, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    max_diff = compare_matrices(m, n, matrix_C_host_cpu, matrix_C_host_gpu);
    printf("shared_mm_sgemm_v3_skip time: %lf , max_diff: %f\n", iElaps, max_diff);

    //先规定好每个block在每个维度处理多少个数(从目标矩阵看)
    constexpr int M_NUM_PER_BLOCK = 32;
    constexpr int N_NUM_PER_BLOCK = 32;
    constexpr int K_NUM_PER_BLOCK = 32;
    
    constexpr int NUM_PER_THREAD = 4; //每个thread输出几个结果
    constexpr size_t ROW_SIZE = 8;   //每一行可以按照float4读数，所以是32的四分之一
    constexpr size_t COL_SIZE = 32;
    dim3 block_float4(ROW_SIZE, COL_SIZE);
    
    //二维网格，分别也和m、n的大小有关
    //只使用原来的四分之一block
    dim3 grid_float4((m + M_NUM_PER_BLOCK - 1) / M_NUM_PER_BLOCK, (n + N_NUM_PER_BLOCK - 1) / N_NUM_PER_BLOCK);
    cudaMemset(matrix_C_device, 0, mem_size_C);
    memset(matrix_C_host_gpu, 0, mem_size_C);
    iStart = cpuSecond();
    shared_mm_sgemm_float4<M_NUM_PER_BLOCK, N_NUM_PER_BLOCK, K_NUM_PER_BLOCK, NUM_PER_THREAD><<<grid_float4, block_float4>>>(matrix_A_device, matrix_B_device, matrix_C_device, m, n, k);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    cudaMemcpy(matrix_C_host_gpu, matrix_C_device, mem_size_C, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    max_diff = compare_matrices(m, n, matrix_C_host_cpu, matrix_C_host_gpu);
    printf("shared_mm_sgemm_float4 time: %lf , max_diff: %f\n", iElaps, max_diff);


    cudaMemset(matrix_C_device, 0, mem_size_C);
    memset(matrix_C_host_gpu, 0, mem_size_C);
    iStart = cpuSecond();
    shared_mm_sgemm_float4_register<M_NUM_PER_BLOCK, N_NUM_PER_BLOCK, K_NUM_PER_BLOCK, NUM_PER_THREAD><<<grid_float4, block_float4>>>(matrix_A_device, matrix_B_device, matrix_C_device, m, n, k);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    cudaMemcpy(matrix_C_host_gpu, matrix_C_device, mem_size_C, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    max_diff = compare_matrices(m, n, matrix_C_host_cpu, matrix_C_host_gpu);
    printf("shared_mm_sgemm_float4_register time: %lf , max_diff: %f\n", iElaps, max_diff);

    free(matrix_A_host);
    free(matrix_B_host);
    free(matrix_C_host_cpu);
    free(matrix_C_host_gpu);

    cudaFree(matrix_A_device);
    cudaFree(matrix_B_device);
    cudaFree(matrix_C_device);
}