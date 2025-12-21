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
        Perf perf("transpose_naive_32*8");
        transpose_naive<<<grid1, block1>>>(d_input, d_output, M, N);
    }

    dim3 block2(16, 16); //每个block线程数最大1024
    //注意这里第一个参数是横向的
    dim3 grid2((N + 16 - 1) / 16, (M + 16 - 1) / 16);
    //先grid，后block
    {
        Perf perf("transpose_naive_16*16");
        transpose_naive<<<grid2, block2>>>(d_input, d_output, M, N);
    }

    dim3 block3(8, 32); //每个block线程数最大1024
    //注意这里第一个参数是横向的
    dim3 grid3((N + 8 - 1) / 8, (M + 32 - 1) / 32);
    //先grid，后block
    {
        Perf perf("transpose_naive_8*32");
        transpose_naive<<<grid3, block3>>>(d_input, d_output, M, N);
    }

    dim3 block_float4(32, 8); //每个block线程数最大1024
    //因为每个线程处理4*4个数，故每个维度线程数要除以4
    dim3 grid_float4(((N >> 2) + block_float4.x - 1) / block_float4.x, ((M >> 2) + block_float4.y - 1) / block_float4.y);

    {
        Perf perf("transpose_float4_32*8");
        transpose_float4<4, 4><<<grid_float4, block_float4>>>(d_input, d_output, M, N);
    }

    dim3 block_float4_2(16, 16); //每个block线程数最大1024
    //因为每个线程处理4*4个数，故每个维度线程数要除以4
    dim3 grid_float4_2(((N >> 2) + block_float4_2.x - 1) / block_float4_2.x, ((M >> 2) + block_float4_2.y - 1) / block_float4_2.y);

    {
        Perf perf("transpose_float4_16*16");
        transpose_float4<4, 4><<<grid_float4_2, block_float4_2>>>(d_input, d_output, M, N);
    }
    

    cudaDeviceSynchronize();


    cudaMemcpy(output_host_gpu, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    for(int i = 0; i < size; i++){
        if(abs(output_host_gpu[i] - output_host_cpu[i]) > 0.0001){
            printf("index [%d] is mismatch, cpu[%f] gpu[%f]\n", i, output_host_cpu[i], output_host_gpu[i]);
            return 0;
        }
    }

    printf("test pass\n");
    return 0;
    
}