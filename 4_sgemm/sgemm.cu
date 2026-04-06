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

//计算模式由内积转为外积
template<unsigned int M_NUM_PER_BLOCK, 
        unsigned int N_NUM_PER_BLOCK, 
        unsigned int K_NUM_PER_BLOCK, 
        unsigned int NUM_PER_THREAD>
__global__ void outer_sgemm_register(float* A, float* B, float* C, const int M, const int N, const int K){
    //每个block负责数据的单个维度
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    //计算全局线程索引，然后重排成ctx * cty的方block
    int tid = ty * blockDim.x + tx;
    int ctx = tid % 16; //32 * 8排成方阵就是16*16
    int cty = tid / 16;
    float* A_ptr_start = A + M_NUM_PER_BLOCK * blockIdx.y * K;
    float* B_ptr_start = B + N_NUM_PER_BLOCK * blockIdx.x;

    __shared__ float a_shared[M_NUM_PER_BLOCK][K_NUM_PER_BLOCK + 4];
    __shared__ float b_shared[K_NUM_PER_BLOCK][N_NUM_PER_BLOCK + 4];
    
    //每个线程依然负责计算NUM_PER_THREAD个数，故每个维度要除以2
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
            a_reg[0] = a_shared[cty * REG_NUM][k];
            a_reg[1] = a_shared[cty * REG_NUM + 1][k];
            b_reg[0] = b_shared[k][ctx * REG_NUM];
            b_reg[1] = b_shared[k][ctx * REG_NUM + 1];
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
            c_ptr_start[(cty * REG_NUM + i) * N + ctx * REG_NUM + j] = temp[i][j];
        }    
    }
    
}


//使用float4加载数据
template<unsigned int M_NUM_PER_BLOCK, 
        unsigned int N_NUM_PER_BLOCK, 
        unsigned int K_NUM_PER_BLOCK, 
        unsigned int M_NUM_PER_THREAD,
        unsigned int N_NUM_PER_THREAD>
__global__ void outer_sgemm_register_float4(float* A, float* B, float* C, const int M, const int N, const int K){
    //每个block负责数据的单个维度
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    float* A_ptr_start = A + M_NUM_PER_BLOCK * blockIdx.y * K;
    float* B_ptr_start = B + N_NUM_PER_BLOCK * blockIdx.x;

    __shared__ float a_shared[M_NUM_PER_BLOCK][K_NUM_PER_BLOCK];
    __shared__ float b_shared[K_NUM_PER_BLOCK][N_NUM_PER_BLOCK];
    
    //每个线程依然负责计算NUM_PER_THREAD个数
    float temp[M_NUM_PER_THREAD][N_NUM_PER_THREAD] = {0.f};
    float a_reg[M_NUM_PER_THREAD] = {0.f};
    float b_reg[N_NUM_PER_THREAD] = {0.f};
    
    int K_NUM_PER_THREAD = 4; //这个大小是由float4指令决定的
    for(int s = 0; s < K; s += K_NUM_PER_BLOCK){
        //每次取4个数，要取4次，因为A的取次数要与B的一次取得数相匹配，都为4
        for(int i = 0; i < M_NUM_PER_THREAD; i++){
            FETCH_FLOAT4(a_shared[ty * M_NUM_PER_THREAD + i][tx * K_NUM_PER_THREAD]) = FETCH_FLOAT4(A_ptr_start[(ty * M_NUM_PER_THREAD + i) * K + s + tx * K_NUM_PER_THREAD]); //在y方向，每个线程负责加载M_NUM_PER_THREAD行数据，所以要乘这个值
        }
        for(int i = 0; i < K_NUM_PER_THREAD; i++){
            FETCH_FLOAT4(b_shared[ty * K_NUM_PER_THREAD + i][tx * N_NUM_PER_THREAD]) = FETCH_FLOAT4(B_ptr_start[(ty * K_NUM_PER_THREAD + s + i) * N + tx * N_NUM_PER_THREAD]);
        }
        
            
        __syncthreads();
        //计算小矩阵之间的mm
        for(int k = 0; k < K_NUM_PER_BLOCK; k++){
            //先把固定k的位置的数加载到寄存器
            //一共64*64个数，16*16个线程。
            //每个线程负责计算16个数，x、y方向各加载4个数
            //A取列，不能使用float4取数
            a_reg[0] = a_shared[ty * M_NUM_PER_THREAD + 0][k];
            a_reg[1] = a_shared[ty * M_NUM_PER_THREAD + 1][k];
            a_reg[2] = a_shared[ty * M_NUM_PER_THREAD + 2][k];
            a_reg[3] = a_shared[ty * M_NUM_PER_THREAD + 3][k];
            //B取行，可以使用float4取数
            FETCH_FLOAT4(b_reg[0]) = FETCH_FLOAT4(b_shared[k][tx * N_NUM_PER_THREAD]);
            
            for(int i = 0; i < M_NUM_PER_THREAD; i++){
                for(int j = 0; j < N_NUM_PER_THREAD; j++){
                    temp[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }
        
        __syncthreads();
    }
    float* c_ptr_start = C + blockIdx.y * N * M_NUM_PER_BLOCK + blockIdx.x * N_NUM_PER_BLOCK;
    // for(int i = 0; i < M_NUM_PER_THREAD; i++){
    //     for(int j = 0; j < N_NUM_PER_THREAD; j++){
    //         //每个维度考虑全局block偏移+block内偏移
    //         //每个thread将负责计算的16个数写回全局内存
    //         //block偏移已经算好了，计算block内偏移即可
    //         c_ptr_start[(ty * M_NUM_PER_THREAD + i) * N + tx * N_NUM_PER_THREAD + j] = temp[i][j];
    //     }    
    // }
    //写入也可以使用float4
    for(int i = 0; i < M_NUM_PER_THREAD; i++){
        FETCH_FLOAT4(c_ptr_start[(ty * M_NUM_PER_THREAD + i) * N + tx * N_NUM_PER_THREAD]) = FETCH_FLOAT4(temp[i][0]);
    }
}
//对A做转置后存储到smem
template<unsigned int M_NUM_PER_BLOCK, 
        unsigned int N_NUM_PER_BLOCK, 
        unsigned int K_NUM_PER_BLOCK, 
        unsigned int M_NUM_PER_THREAD,
        unsigned int N_NUM_PER_THREAD>
__global__ void outer_sgemm_register_float4_transpose(float* A, float* B, float* C, const int M, const int N, const int K){
    //每个block负责数据的单个维度
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    float* A_ptr_start = A + M_NUM_PER_BLOCK * blockIdx.y * K;
    float* B_ptr_start = B + N_NUM_PER_BLOCK * blockIdx.x;
    //用于load数据时暂存做转置,大小为K_NUM_PER_THREAD
    float a_reg_load[4] = {0.f};

    __shared__ float a_shared[K_NUM_PER_BLOCK][M_NUM_PER_BLOCK];
    __shared__ float b_shared[K_NUM_PER_BLOCK][N_NUM_PER_BLOCK];
    
    //每个线程依然负责计算NUM_PER_THREAD个数
    float temp[M_NUM_PER_THREAD][N_NUM_PER_THREAD] = {0.f};
    float a_reg[M_NUM_PER_THREAD] = {0.f};
    float b_reg[N_NUM_PER_THREAD] = {0.f};
    
    int K_NUM_PER_THREAD = 4; //这个大小是由float4指令决定的
    for(int s = 0; s < K; s += K_NUM_PER_BLOCK){
        //每次取4个数，要取4次，因为A的取次数要与B的一次取得数相匹配，都为4
        for(int i = 0; i < M_NUM_PER_THREAD; i++){
            //先把数据加载到寄存器中
            FETCH_FLOAT4(a_reg_load[0]) = FETCH_FLOAT4(A_ptr_start[(ty * M_NUM_PER_THREAD + i) * K + s + tx * K_NUM_PER_THREAD]); //在y方向，每个线程负责加载M_NUM_PER_THREAD行数据，所以要乘这个值
            //在global内存中，tx对应K维，ty对应M维;在转置到smem后，两维索引需要交换位置，但计算方式不变
            a_shared[tx * K_NUM_PER_THREAD + 0][ty * M_NUM_PER_THREAD + i] = a_reg_load[0];
            a_shared[tx * K_NUM_PER_THREAD + 1][ty * M_NUM_PER_THREAD + i] = a_reg_load[1];
            a_shared[tx * K_NUM_PER_THREAD + 2][ty * M_NUM_PER_THREAD + i] = a_reg_load[2];
            a_shared[tx * K_NUM_PER_THREAD + 3][ty * M_NUM_PER_THREAD + i] = a_reg_load[3];
        }
        for(int i = 0; i < K_NUM_PER_THREAD; i++){
            FETCH_FLOAT4(b_shared[ty * K_NUM_PER_THREAD + i][tx * N_NUM_PER_THREAD]) = FETCH_FLOAT4(B_ptr_start[(ty * K_NUM_PER_THREAD + s + i) * N + tx * N_NUM_PER_THREAD]);
        }
        
            
        __syncthreads();
        //计算小矩阵之间的mm
        for(int k = 0; k < K_NUM_PER_BLOCK; k++){
            //先把固定k的位置的数加载到寄存器
            //一共64*64个数，16*16个线程。
            //每个线程负责计算16个数，x、y方向各加载4个数
            //计算时需要看目标矩阵的tile
            FETCH_FLOAT4(a_reg[0]) = FETCH_FLOAT4(a_shared[k][ty * M_NUM_PER_THREAD]);
            //B取行，可以使用float4取数
            FETCH_FLOAT4(b_reg[0]) = FETCH_FLOAT4(b_shared[k][tx * N_NUM_PER_THREAD]);
            
            for(int i = 0; i < M_NUM_PER_THREAD; i++){
                for(int j = 0; j < N_NUM_PER_THREAD; j++){
                    temp[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }
        
        __syncthreads();
    }
    float* c_ptr_start = C + blockIdx.y * N * M_NUM_PER_BLOCK + blockIdx.x * N_NUM_PER_BLOCK;
    // for(int i = 0; i < M_NUM_PER_THREAD; i++){
    //     for(int j = 0; j < N_NUM_PER_THREAD; j++){
    //         //每个维度考虑全局block偏移+block内偏移
    //         //每个thread将负责计算的16个数写回全局内存
    //         //block偏移已经算好了，计算block内偏移即可
    //         c_ptr_start[(ty * M_NUM_PER_THREAD + i) * N + tx * N_NUM_PER_THREAD + j] = temp[i][j];
    //     }    
    // }
    //写入也可以使用float4
    for(int i = 0; i < M_NUM_PER_THREAD; i++){
        FETCH_FLOAT4(c_ptr_start[(ty * M_NUM_PER_THREAD + i) * N + tx * N_NUM_PER_THREAD]) = FETCH_FLOAT4(temp[i][0]);
    }
}

//对A做转置后存储到smem
template<unsigned int M_NUM_PER_BLOCK, 
        unsigned int N_NUM_PER_BLOCK, 
        unsigned int K_NUM_PER_BLOCK, 
        unsigned int M_NUM_PER_THREAD,
        unsigned int N_NUM_PER_THREAD>
__global__ void sgemm_pingpong(float* A, float* B, float* C, const int M, const int N, const int K){
    //每个block负责数据的单个维度
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    const int tid = ty * blockDim.x + tx;  //拿到block内全局索引

    __shared__ float a_shared[2][K_NUM_PER_BLOCK][M_NUM_PER_BLOCK];
    __shared__ float b_shared[2][K_NUM_PER_BLOCK][N_NUM_PER_BLOCK];

    //每个线程依然负责计算NUM_PER_THREAD个数
    float temp[M_NUM_PER_THREAD][N_NUM_PER_THREAD] = {0.f};
    float a_reg[M_NUM_PER_THREAD] = {0.f};
    float b_reg[N_NUM_PER_THREAD] = {0.f};
    //用于load数据时暂存做转置,大小为K_NUM_PER_THREAD
    float a_reg_load[4] = {0.f};
    
    float* A_ptr_start = A + M_NUM_PER_BLOCK * blockIdx.y * K;
    float* B_ptr_start = B + N_NUM_PER_BLOCK * blockIdx.x;

    const int A_tile_thread_per_row = K_NUM_PER_BLOCK / 4;// 8 // 4 = 2,4是float4的字节数
    const int B_tile_thread_per_row = N_NUM_PER_BLOCK / 4;// 128 // 4 = 32,4是float4的字节数
    //对block全局线程id做一个重排，A和B的buffer形状不同
    const int A_tile_tid_x = tid % A_tile_thread_per_row;  //取值0-1
    const int A_tile_tid_y = tid / A_tile_thread_per_row;  //取值 0-127

    const int B_tile_tid_x = tid % B_tile_thread_per_row;  //取值 0-31
    const int B_tile_tid_y = tid / B_tile_thread_per_row;  //取值 0-7

    int K_NUM_PER_THREAD = 4; //这个大小是由float4指令决定的

    //取第一块内存
    FETCH_FLOAT4(a_reg_load[0]) = FETCH_FLOAT4(A_ptr_start[A_tile_tid_y * K + A_tile_tid_x * K_NUM_PER_THREAD]);
    //做了转置，所以列id都是一样的
    a_shared[0][A_tile_tid_x * K_NUM_PER_THREAD + 0][A_tile_tid_y] = a_reg_load[0];
    a_shared[0][A_tile_tid_x * K_NUM_PER_THREAD + 1][A_tile_tid_y] = a_reg_load[1];
    a_shared[0][A_tile_tid_x * K_NUM_PER_THREAD + 2][A_tile_tid_y] = a_reg_load[2];
    a_shared[0][A_tile_tid_x * K_NUM_PER_THREAD + 3][A_tile_tid_y] = a_reg_load[3];

    FETCH_FLOAT4(b_shared[0][B_tile_tid_y][B_tile_tid_x * K_NUM_PER_THREAD]) = FETCH_FLOAT4(B_ptr_start[B_tile_tid_y * N + B_tile_tid_x * K_NUM_PER_THREAD]);
    __syncthreads();
    int write_buffer_idx = 1;
    for(int s = K_NUM_PER_BLOCK; s < K; s += K_NUM_PER_BLOCK){
        //继续取第二块buffer的内存,和之前一样，只是需要在K方向多偏移一个s
        FETCH_FLOAT4(a_reg_load[0]) = FETCH_FLOAT4(A_ptr_start[A_tile_tid_y * K + A_tile_tid_x * K_NUM_PER_THREAD + s]);
        //做了转置，所以列id都是一样的
        a_shared[write_buffer_idx][A_tile_tid_x * K_NUM_PER_THREAD + 0][A_tile_tid_y] = a_reg_load[0];
        a_shared[write_buffer_idx][A_tile_tid_x * K_NUM_PER_THREAD + 1][A_tile_tid_y] = a_reg_load[1];
        a_shared[write_buffer_idx][A_tile_tid_x * K_NUM_PER_THREAD + 2][A_tile_tid_y] = a_reg_load[2];
        a_shared[write_buffer_idx][A_tile_tid_x * K_NUM_PER_THREAD + 3][A_tile_tid_y] = a_reg_load[3];

        FETCH_FLOAT4(b_shared[write_buffer_idx][B_tile_tid_y][B_tile_tid_x * K_NUM_PER_THREAD]) = FETCH_FLOAT4(B_ptr_start[(B_tile_tid_y + s) * N + B_tile_tid_x * K_NUM_PER_THREAD]);
        //改变当前流水级别buffer idx
        write_buffer_idx ^= 1;
        //计算小矩阵之间的mm
        for(int k = 0; k < K_NUM_PER_BLOCK; k++){
            //计算时需要看目标矩阵的tile
            FETCH_FLOAT4(a_reg[0]) = FETCH_FLOAT4(a_shared[write_buffer_idx][k][ty * M_NUM_PER_THREAD]);
            FETCH_FLOAT4(a_reg[4]) = FETCH_FLOAT4(a_shared[write_buffer_idx][k][ty * M_NUM_PER_THREAD + 4]);
            //B取行，可以使用float4取数
            FETCH_FLOAT4(b_reg[0]) = FETCH_FLOAT4(b_shared[write_buffer_idx][k][tx * N_NUM_PER_THREAD]);
            FETCH_FLOAT4(b_reg[4]) = FETCH_FLOAT4(b_shared[write_buffer_idx][k][tx * N_NUM_PER_THREAD + 4]);
            
            for(int i = 0; i < M_NUM_PER_THREAD; i++){
                for(int j = 0; j < N_NUM_PER_THREAD; j++){
                    temp[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }
        
        __syncthreads();
    }
    write_buffer_idx ^= 1;
    //计算最后一个小矩阵之间的mm
    for(int k = 0; k < K_NUM_PER_BLOCK; k++){
        //计算时需要看目标矩阵的tile
        FETCH_FLOAT4(a_reg[0]) = FETCH_FLOAT4(a_shared[write_buffer_idx][k][ty * M_NUM_PER_THREAD]);
        FETCH_FLOAT4(a_reg[4]) = FETCH_FLOAT4(a_shared[write_buffer_idx][k][ty * M_NUM_PER_THREAD + 4]);
        //B取行，可以使用float4取数
        FETCH_FLOAT4(b_reg[0]) = FETCH_FLOAT4(b_shared[write_buffer_idx][k][tx * N_NUM_PER_THREAD]);
        FETCH_FLOAT4(b_reg[4]) = FETCH_FLOAT4(b_shared[write_buffer_idx][k][tx * N_NUM_PER_THREAD + 4]);
        
        for(int i = 0; i < M_NUM_PER_THREAD; i++){
            for(int j = 0; j < N_NUM_PER_THREAD; j++){
                temp[i][j] += a_reg[i] * b_reg[j];
            }
        }
    }
    __syncthreads();

    float* c_ptr_start = C + blockIdx.y * N * M_NUM_PER_BLOCK + blockIdx.x * N_NUM_PER_BLOCK;
    // for(int i = 0; i < M_NUM_PER_THREAD; i++){
    //     for(int j = 0; j < N_NUM_PER_THREAD; j++){
    //         //每个维度考虑全局block偏移+block内偏移
    //         //每个thread将负责计算的16个数写回全局内存
    //         //block偏移已经算好了，计算block内偏移即可
    //         c_ptr_start[(ty * M_NUM_PER_THREAD + i) * N + tx * N_NUM_PER_THREAD + j] = temp[i][j];
    //     }    
    // }
    //写入也可以使用float4
    for(int i = 0; i < M_NUM_PER_THREAD; i++){
        FETCH_FLOAT4(c_ptr_start[(ty * M_NUM_PER_THREAD + i) * N + tx * N_NUM_PER_THREAD]) = FETCH_FLOAT4(temp[i][0]);
        FETCH_FLOAT4(c_ptr_start[(ty * M_NUM_PER_THREAD + i) * N + tx * N_NUM_PER_THREAD + 4]) = FETCH_FLOAT4(temp[i][4]);
    }
}

int main(){
    double iStart, iElaps;

    int m = 2048;
    int n = 2048;
    constexpr int k = 2048;
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
    outer_sgemm_register<M_NUM_PER_BLOCK, N_NUM_PER_BLOCK, K_NUM_PER_BLOCK, NUM_PER_THREAD><<<grid_float4, block_float4>>>(matrix_A_device, matrix_B_device, matrix_C_device, m, n, k);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    cudaMemcpy(matrix_C_host_gpu, matrix_C_device, mem_size_C, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    max_diff = compare_matrices(m, n, matrix_C_host_cpu, matrix_C_host_gpu);
    printf("outer_sgemm_register time: %lf , max_diff: %f\n", iElaps, max_diff);

    //先规定好每个block在每个维度处理多少个数(从目标矩阵看)
    constexpr int M_NUM_PER_BLOCK_OUTER = 64;
    constexpr int N_NUM_PER_BLOCK_OUTER = 64;
    constexpr int K_NUM_PER_BLOCK_OUTER = 64;
    
    constexpr int NUM_PER_THREAD_OUTER = 16; //每个thread输出几个结果
    constexpr int M_NUM_PER_THREAD_OUTER = 4;  //每个线程在每个维度计算几个数，两个乘积应该是NUM_PER_THREAD_OUTER
    constexpr int N_NUM_PER_THREAD_OUTER = 4;
    constexpr size_t ROW_SIZE_OUTER = 16;   //每一行可以按照float4读数，所以是64的四分之一
    constexpr size_t COL_SIZE_OUTER = 16;
    dim3 block_float4_outer(ROW_SIZE_OUTER, COL_SIZE_OUTER);
    
    //二维网格，分别也和m、n的大小有关
    //只使用原来的四分之一block
    dim3 grid_float4_outer((m + M_NUM_PER_BLOCK_OUTER - 1) / M_NUM_PER_BLOCK_OUTER, (n + N_NUM_PER_BLOCK_OUTER - 1) / N_NUM_PER_BLOCK_OUTER);
    cudaMemset(matrix_C_device, 0, mem_size_C);
    memset(matrix_C_host_gpu, 0, mem_size_C);
    iStart = cpuSecond();
    outer_sgemm_register_float4<M_NUM_PER_BLOCK_OUTER, N_NUM_PER_BLOCK_OUTER, K_NUM_PER_BLOCK_OUTER, M_NUM_PER_THREAD_OUTER, N_NUM_PER_THREAD_OUTER><<<grid_float4_outer, block_float4_outer>>>(matrix_A_device, matrix_B_device, matrix_C_device, m, n, k);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    cudaMemcpy(matrix_C_host_gpu, matrix_C_device, mem_size_C, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    max_diff = compare_matrices(m, n, matrix_C_host_cpu, matrix_C_host_gpu);
    printf("outer_sgemm_register_float4 time: %lf , max_diff: %f\n", iElaps, max_diff);

    cudaMemset(matrix_C_device, 0, mem_size_C);
    memset(matrix_C_host_gpu, 0, mem_size_C);
    iStart = cpuSecond();
    outer_sgemm_register_float4_transpose<M_NUM_PER_BLOCK_OUTER, N_NUM_PER_BLOCK_OUTER, K_NUM_PER_BLOCK_OUTER, M_NUM_PER_THREAD_OUTER, N_NUM_PER_THREAD_OUTER><<<grid_float4_outer, block_float4_outer>>>(matrix_A_device, matrix_B_device, matrix_C_device, m, n, k);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    cudaMemcpy(matrix_C_host_gpu, matrix_C_device, mem_size_C, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    max_diff = compare_matrices(m, n, matrix_C_host_cpu, matrix_C_host_gpu);
    printf("outer_sgemm_register_float4_transpose time: %lf , max_diff: %f\n", iElaps, max_diff);


    //先规定好每个block在每个维度处理多少个数(从目标矩阵看)
    constexpr int M_NUM_PER_BLOCK_PINGPONG = 128;
    constexpr int N_NUM_PER_BLOCK_PINGPONG = 128;
    constexpr int K_NUM_PER_BLOCK_PINGPONG = 8;
    
    constexpr int NUM_PER_THREAD_PINGPONG = 64; //每个thread输出几个结果
    constexpr int M_NUM_PER_THREAD_PINGPONG = 8;  //每个线程在每个维度计算几个数，两个乘积应该是NUM_PER_THREAD_OUTER
    constexpr int N_NUM_PER_THREAD_PINGPONG = 8;
    constexpr size_t ROW_SIZE_PINGPONG = M_NUM_PER_BLOCK_PINGPONG / M_NUM_PER_THREAD_PINGPONG;   
    constexpr size_t COL_SIZE_PINGPONG = N_NUM_PER_BLOCK_PINGPONG / N_NUM_PER_THREAD_PINGPONG;
    dim3 block_pingpong(ROW_SIZE_PINGPONG, COL_SIZE_PINGPONG);
    
    //二维网格，分别也和m、n的大小有关
    //只使用原来的四分之一block
    //顺序为先x后y
    dim3 grid_pingpong((n + N_NUM_PER_BLOCK_PINGPONG - 1) / N_NUM_PER_BLOCK_PINGPONG, (m + M_NUM_PER_BLOCK_PINGPONG - 1) / M_NUM_PER_BLOCK_PINGPONG);
    cudaMemset(matrix_C_device, 0, mem_size_C);
    memset(matrix_C_host_gpu, 0, mem_size_C);
    iStart = cpuSecond();
    sgemm_pingpong<M_NUM_PER_BLOCK_PINGPONG, N_NUM_PER_BLOCK_PINGPONG, K_NUM_PER_BLOCK_PINGPONG, M_NUM_PER_THREAD_PINGPONG, N_NUM_PER_THREAD_PINGPONG><<<grid_pingpong, block_pingpong>>>(matrix_A_device, matrix_B_device, matrix_C_device, m, n, k);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    cudaMemcpy(matrix_C_host_gpu, matrix_C_device, mem_size_C, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    max_diff = compare_matrices(m, n, matrix_C_host_cpu, matrix_C_host_gpu);
    printf("sgemm_pingpong time: %lf , max_diff: %f\n", iElaps, max_diff);
    
    
    free(matrix_A_host);
    free(matrix_B_host);
    free(matrix_C_host_cpu);
    free(matrix_C_host_gpu);

    cudaFree(matrix_A_device);
    cudaFree(matrix_B_device);
    cudaFree(matrix_C_device);
}