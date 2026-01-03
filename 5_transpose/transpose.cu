#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <iostream>

class Perf{
public:
    Perf(const std::string &name)
    : m_name(name)
    {
        cudaEventCreate(&m_start);
        cudaEventCreate(&m_end);
        cudaEventRecord(m_start);
        cudaEventSynchronize(m_start);
    }
    ~Perf(){
        cudaEventRecord(m_end);
        cudaEventSynchronize(m_end);
        float elapsed_time = 0.0;
        cudaEventElapsedTime(&elapsed_time, m_start, m_end);
        std::cout << m_name << " elapse: " << elapsed_time << " ms" << std::endl;
    }
private:
    std::string m_name;
    cudaEvent_t m_start, m_end;
};

bool check_result(float* device_output, float* cpu_output, size_t size){
    cudaDeviceSynchronize();

    float* output_host_gpu = (float*)malloc(size * sizeof(float));
    cudaMemcpy(output_host_gpu, device_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    for(int i = 0; i < size; i++){
        if(abs(output_host_gpu[i] - cpu_output[i]) > 0.0001){
            printf("index [%d] is mismatch, cpu[%f] gpu[%f]\n", i, cpu_output[i], output_host_gpu[i]);
            return false;
        }
    }

    free(output_host_gpu);

    return true;
}

void cpu_transpose(float* input, float* out, const int M, const int N){
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            int ori_idx = i * N + j;
            int new_idx = j * M + i;
            out[new_idx] = input[ori_idx];
        }
    }
}

__global__ void transpose_naive(float* input, float* output, const int M, const int N){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int ori_idx = y * N + x;
    int new_idx = x * M + y;
    output[new_idx] = input[ori_idx];
}

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&pointer)[0])
template<const int PER_THREAD_M, const int PER_THREAD_N>
__global__ void transpose_float4(float* input, float* output, const int M, const int N){
    // M * N，M为行数在y维
    float src[PER_THREAD_M][PER_THREAD_N];
    float dst[PER_THREAD_M][PER_THREAD_N];

    float* input_start = input + blockIdx.y * PER_THREAD_M * blockDim.y * N + blockIdx.x * PER_THREAD_N * blockDim.x;
    for(int i = 0; i < PER_THREAD_M; i++){
        FETCH_FLOAT4(src[i][0]) = FETCH_FLOAT4(input_start[(threadIdx.y * PER_THREAD_M + i) * N + threadIdx.x * PER_THREAD_N]);
    }

    FETCH_FLOAT4(dst[0]) = make_float4(src[0][0], src[1][0], src[2][0], src[3][0]);
    FETCH_FLOAT4(dst[1]) = make_float4(src[0][1], src[1][1], src[2][1], src[3][1]);
    FETCH_FLOAT4(dst[2]) = make_float4(src[0][2], src[1][2], src[2][2], src[3][2]);
    FETCH_FLOAT4(dst[3]) = make_float4(src[0][3], src[1][3], src[2][3], src[3][3]);

    //与input相比，就是把之前的索引(i, j)->(j, i)，要分清哪个是i，哪个是j
    float* out = output + blockIdx.x * PER_THREAD_N * blockDim.x * M + blockIdx.y * PER_THREAD_M * blockDim.y;
    //对于block内，threadIdx.y * PER_THREAD_M为i, threadIdx.x * PER_THREAD_M为j
    //定位到thread负责数据块的左上端点即可，因为块内数据已经处理完了
    for(int i = 0; i < PER_THREAD_N; i++){
        FETCH_FLOAT4(out[(threadIdx.x * PER_THREAD_N + i) * M + threadIdx.y * PER_THREAD_M]) = FETCH_FLOAT4(dst[i][0]);
    }
}

#define FETCH_FLOAT2(pointer) (reinterpret_cast<float2*>(&pointer)[0])
template<const int PER_THREAD_M, const int PER_THREAD_N>
__global__ void transpose_float2(float* input, float* output, const int M, const int N){
    // M * N，M为行数在y维
    float src[PER_THREAD_M][PER_THREAD_N];
    float dst[PER_THREAD_M][PER_THREAD_N];

    float* input_start = input + blockIdx.y * PER_THREAD_M * blockDim.y * N + blockIdx.x * PER_THREAD_N * blockDim.x;
    for(int i = 0; i < PER_THREAD_M; i++){
        FETCH_FLOAT2(src[i][0]) = FETCH_FLOAT2(input_start[(threadIdx.y * PER_THREAD_M + i) * N + threadIdx.x * PER_THREAD_N]);
    }

    FETCH_FLOAT2(dst[0]) = make_float2(src[0][0], src[1][0]);
    FETCH_FLOAT2(dst[1]) = make_float2(src[0][1], src[1][1]);

    //与input相比，就是把之前的索引(i, j)->(j, i)，要分清哪个是i，哪个是j
    float* out = output + blockIdx.x * PER_THREAD_N * blockDim.x * M + blockIdx.y * PER_THREAD_M * blockDim.y;
    //对于block内，threadIdx.y * PER_THREAD_M为i, threadIdx.x * PER_THREAD_M为j
    //定位到thread负责数据块的左上端点即可，因为块内数据已经处理完了
    for(int i = 0; i < PER_THREAD_N; i++){
        FETCH_FLOAT2(out[(threadIdx.x * PER_THREAD_N + i) * M + threadIdx.y * PER_THREAD_M]) = FETCH_FLOAT2(dst[i][0]);
    }
}

template<const int PER_THREAD_M, const int PER_THREAD_N>
__global__ void transpose_float2_1x2(float* input, float* output, const int M, const int N){
    // M * N，M为行数在y维
    float src[PER_THREAD_M];
    float dst[PER_THREAD_M];

    float* input_start = input + blockIdx.y * PER_THREAD_M * blockDim.y * N + blockIdx.x * PER_THREAD_N * blockDim.x;
    for(int i = 0; i < PER_THREAD_M; i++){
        src[i] = input_start[(threadIdx.y * PER_THREAD_M + i) * N + threadIdx.x * PER_THREAD_N];
    }

    FETCH_FLOAT2(dst[0]) = make_float2(src[0], src[1]);

    //与input相比，就是把之前的索引(i, j)->(j, i)，要分清哪个是i，哪个是j
    float* out = output + blockIdx.x * PER_THREAD_N * blockDim.x * M + blockIdx.y * PER_THREAD_M * blockDim.y;
    //对于block内，threadIdx.y * PER_THREAD_M为i, threadIdx.x * PER_THREAD_M为j
    //定位到thread负责数据块的左上端点即可，因为块内数据已经处理完了
    
    FETCH_FLOAT2(out[(threadIdx.x * PER_THREAD_N) * M + threadIdx.y * PER_THREAD_M]) = FETCH_FLOAT2(dst[0]);
    
}

template<const int BLOCK_SIZE>
__global__ void transpose_shared_mem(float* input, float* output, const int M, const int N){
    __shared__ float sdata[BLOCK_SIZE][BLOCK_SIZE];
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int m = blockDim.y * blockIdx.y + threadIdx.y;
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    float* input_start = input + blockIdx.y * blockDim.y * N + blockIdx.x * blockDim.x;
    
    if(m < M && n < N){
        //转置读到shared memory中
        sdata[tx][ty] = input_start[tx + ty * N];  //这里x和y的索引前后是交换的
    }
    __syncthreads();
    //在block中，x维对应的是N，y维对应的是M
    //转置以后还是这么对应
    m = blockDim.y * blockIdx.y + threadIdx.x;
    n = blockDim.x * blockIdx.x + threadIdx.y;
    //与input相比，就是把之前的索引(i, j)->(j, i)，要分清哪个是i，哪个是j
    float* out = output + blockIdx.x * blockDim.x * M + blockIdx.y * blockDim.y;
    if(m < M && n < N){
        //此时整体内存需要转置，也就是x方向为M，y方向为N
        //但是在block内已经做过转置了,所以仍保留原方向，不需要再做转置
        out[ty * M + tx] = sdata[ty][tx]; //这里x和y的索引前后是对应的
    }
    
}

template<const int BLOCK_SIZE>
__global__ void transpose_shared_mem_padding(float* input, float* output, const int M, const int N){
    __shared__ float sdata[BLOCK_SIZE][BLOCK_SIZE + 1];
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int m = blockDim.y * blockIdx.y + threadIdx.y;
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    float* input_start = input + blockIdx.y * blockDim.y * N + blockIdx.x * blockDim.x;
    
    if(m < M && n < N){
        //转置读到shared memory中
        sdata[tx][ty] = input_start[tx + ty * N];  //这里x和y的索引前后是交换的
    }
    __syncthreads();
    //在block中，x维对应的是N，y维对应的是M
    //转置以后还是这么对应
    m = blockDim.y * blockIdx.y + threadIdx.x;
    n = blockDim.x * blockIdx.x + threadIdx.y;
    //与input相比，就是把之前的索引(i, j)->(j, i)，要分清哪个是i，哪个是j
    float* out = output + blockIdx.x * blockDim.x * M + blockIdx.y * blockDim.y;
    if(m < M && n < N){
        //此时整体内存需要转置，也就是x方向为M，y方向为N
        //但是在block内已经做过转置了,所以仍保留原方向，不需要再做转置
        out[ty * M + tx] = sdata[ty][tx]; //这里x和y的索引前后是对应的
    }
    
}

template<const int BLOCK_SIZE, const int NUM_PER_THREAD>
__global__ void transpose_shared_mem_padding_multi(float* input, float* output, const int M, const int N){
    __shared__ float sdata[BLOCK_SIZE][BLOCK_SIZE + 1];
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int STEP = BLOCK_SIZE / NUM_PER_THREAD;

    int m = blockDim.y * blockIdx.y + threadIdx.y;
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    float* input_start = input + blockIdx.y * blockDim.y * N + blockIdx.x * blockDim.x;
    //在x维度，每个线程处理NUM_PER_THREAD个数
    for(int i = 0; i < NUM_PER_THREAD; i++){
        //每个线程每隔STEP个数取一个数，如果取连续的数会导致访存不能合并
        if(m < M && n + i * STEP < N){
            //转置读到shared memory中
            sdata[tx + i * STEP][ty] = input_start[tx + i * STEP + ty * N];  //这里x和y的索引前后是交换的
        }
    }
    
    __syncthreads();
    //在block中，x维对应的是N，y维对应的是M
    //转置以后还是这么对应
    m = blockDim.y * blockIdx.y + threadIdx.x;
    n = blockDim.x * blockIdx.x + threadIdx.y;
    //与input相比，就是把之前的索引(i, j)->(j, i)，要分清哪个是i，哪个是j
    float* out = output + blockIdx.x * blockDim.x * M + blockIdx.y * blockDim.y;
    for(int i = 0; i < NUM_PER_THREAD; i++){
        if(m + i * STEP < M && n < N){
            //此时整体内存需要转置，也就是x方向为M，y方向为N
            //但是在block内已经做过转置了,所以仍保留原方向，不需要再做转置
            out[ty * M + tx + i * STEP] = sdata[ty][tx + i * STEP]; //这里x和y的索引前后是对应的
        }
    }
    
}



int main(){
    const int M = 2048;
    const int N = 512;
    const size_t size =  M * N;
    float *input_host = (float*)malloc(size * sizeof(float));
    float *output_host_cpu = (float*)malloc(size * sizeof(float));
    float *output_host_gpu = (float*)malloc(size * sizeof(float));

    for(int i = 0; i < size; i++){
        input_host[i] = 2.0 * (float)drand48() - 1.0;
    }
    {
        Perf perf("cpu_transpose");
        cpu_transpose(input_host, output_host_cpu, M, N);
    }

    float *d_input, *d_output;

    cudaMalloc((void **)&d_input, size * sizeof(float));
    cudaMalloc((void **)&d_output, size * sizeof(float));

    cudaMemcpy(d_input, input_host, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0 , size * sizeof(float));

    const int M_NUM = 8;
    const int N_NUM = 32;
    dim3 block1(N_NUM, M_NUM); //每个block线程数最大1024
    //注意这里第一个参数是横向的
    dim3 grid1((N + N_NUM - 1) / N_NUM, (M + M_NUM - 1) / M_NUM);
    //先grid，后block
    {
        cudaMemset(d_output, 0, size);
        Perf perf("transpose_naive_32*8");
        transpose_naive<<<grid1, block1>>>(d_input, d_output, M, N);
    }
    check_result(d_output, output_host_cpu, size);

    dim3 block2(16, 16); //每个block线程数最大1024
    //注意这里第一个参数是横向的
    dim3 grid2((N + 16 - 1) / 16, (M + 16 - 1) / 16);
    //先grid，后block
    {
        cudaMemset(d_output, 0, size);
        Perf perf("transpose_naive_16*16");
        transpose_naive<<<grid2, block2>>>(d_input, d_output, M, N);
    }
    check_result(d_output, output_host_cpu, size);

    dim3 block3(8, 32); //每个block线程数最大1024
    //注意这里第一个参数是横向的
    dim3 grid3((N + 8 - 1) / 8, (M + 32 - 1) / 32);
    //先grid，后block
    {
        cudaMemset(d_output, 0, size);
        Perf perf("transpose_naive_8*32");
        transpose_naive<<<grid3, block3>>>(d_input, d_output, M, N);
    }
    check_result(d_output, output_host_cpu, size);

    dim3 block_float4(32, 8); //每个block线程数最大1024
    //因为每个线程处理4*4个数，故每个维度线程数要除以4
    dim3 grid_float4(((N >> 2) + block_float4.x - 1) / block_float4.x, ((M >> 2) + block_float4.y - 1) / block_float4.y);

    {
        cudaMemset(d_output, 0, size);
        Perf perf("transpose_float4_32*8");
        transpose_float4<4, 4><<<grid_float4, block_float4>>>(d_input, d_output, M, N);
    }
    check_result(d_output, output_host_cpu, size);

    dim3 block_float4_2(16, 16); //每个block线程数最大1024
    //因为每个线程处理4*4个数，故每个维度线程数要除以4
    dim3 grid_float4_2(((N >> 2) + block_float4_2.x - 1) / block_float4_2.x, ((M >> 2) + block_float4_2.y - 1) / block_float4_2.y);

    {
        cudaMemset(d_output, 0, size);
        Perf perf("transpose_float4_16*16");
        transpose_float4<4, 4><<<grid_float4_2, block_float4_2>>>(d_input, d_output, M, N);
    }
    check_result(d_output, output_host_cpu, size);

    dim3 block_float2(16, 16); //每个block线程数最大1024
    //因为每个线程处理2*2个数，故每个维度线程数要除以2
    dim3 grid_float2(((N >> 1) + block_float2.x - 1) / block_float2.x, ((M >> 1) + block_float2.y - 1) / block_float2.y);

    {
        cudaMemset(d_output, 0, size);
        Perf perf("transpose_float2_16*16");
        transpose_float2<2, 2><<<grid_float2, block_float2>>>(d_input, d_output, M, N);
    }
    check_result(d_output, output_host_cpu, size);

    dim3 block_float2_1x2(8, 32); //每个block线程数最大1024
    //因为每个线程处理1*2个数，故M维度线程数要除以2
    dim3 grid_float2_1x2((N + block_float2_1x2.x - 1) / block_float2_1x2.x, ((M >> 1) + block_float2_1x2.y - 1) / block_float2_1x2.y);

    {
        cudaMemset(d_output, 0, size);
        Perf perf("transpose_float2_1x2_8*32");
        transpose_float2_1x2<2, 1><<<grid_float2_1x2, block_float2_1x2>>>(d_input, d_output, M, N);
    }
    check_result(d_output, output_host_cpu, size);

    {
        cudaMemset(d_output, 0, size);
        Perf perf("transpose_shared_mem_16*16");
        transpose_shared_mem<16><<<grid2, block2>>>(d_input, d_output, M, N);
    }
    check_result(d_output, output_host_cpu, size);
    

    {
        cudaMemset(d_output, 0, size);
        Perf perf("transpose_shared_mem_padding_16*16");
        transpose_shared_mem_padding<16><<<grid2, block2>>>(d_input, d_output, M, N);
    }
    check_result(d_output, output_host_cpu, size);

    //在x维缩减实际线程数，但逻辑处理数据数不变
    dim3 block_multi(32 / 4, 32); //每个block线程数最大1024
    //注意这里第一个参数是横向的
    dim3 grid_multi((N + 32 - 1) / 32, (M + 32 - 1) / 32);
    {
        cudaMemset(d_output, 0, size);
        Perf perf("transpose_shared_mem_padding_multi");
        transpose_shared_mem_padding_multi<32, 4><<<grid_multi, block_multi>>>(d_input, d_output, M, N);
    }
    check_result(d_output, output_host_cpu, size);

    

    printf("test pass\n");
    return 0;
    
}