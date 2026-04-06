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

    for(int i = 0; i < size; i++){
        if(abs(device_output[i] - cpu_output[i]) > 0.1){
            printf("index [%d] is mismatch, cpu[%f] gpu[%f]\n", i, cpu_output[i], device_output[i]);
            return false;
        }
    }

    return true;
}

void cpu_prefix_sum(float* input, float* out, const int N){
    out[0] = input[0];
    for(int i = 1; i < N; i++){
        out[i] = out[i - 1] + input[i];
    }
}

template<size_t SECTION_SIZE>
__global__ void kogge_stone_prefix_sum(float* input, float* out, const int N, const int element_stride=1){
    __shared__ float temp_out[SECTION_SIZE];
    uint32_t global_idx = blockIdx.x * blockDim.x + (threadIdx.x * element_stride);
    if(global_idx < N) temp_out[threadIdx.x] = input[global_idx];
    else temp_out[threadIdx.x] = 0.0f;

    for(int stride = 1; stride < SECTION_SIZE; stride <<= 1){
        __syncthreads();
        float val = temp_out[threadIdx.x];
        if (threadIdx.x >= stride) {
            val += temp_out[threadIdx.x - stride];
        }
        __syncthreads();
        temp_out[threadIdx.x] = val;
    }

    if(global_idx < N) out[global_idx] = temp_out[threadIdx.x];

}

#define SET_SECTION_SIZE (512)
void kogge_stone_prefix_sum_host(float* input, float* out, const int N){
    float *d_input, *d_output;

    cudaMalloc((void **)&d_input, N * sizeof(float));
    cudaMalloc((void **)&d_output, N * sizeof(float));

    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0 , N * sizeof(float));

    size_t num_blocks = (N + SET_SECTION_SIZE - 1) / SET_SECTION_SIZE;
    dim3 step1_block(SET_SECTION_SIZE);
    dim3 step2_grid(num_blocks);
    kogge_stone_prefix_sum<SET_SECTION_SIZE><<<step2_grid, step1_block>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(out, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    float last_block_sum = 0;
    for(int idx = 0; idx < N; idx++){
        if(idx % SET_SECTION_SIZE == 0 && idx != 0){
            last_block_sum = out[idx - 1];
        }
        out[idx] += last_block_sum;
    }
}

template<size_t SECTION_SIZE>
__global__ void thread_coarsening_prefix_sum(float* input, float* out, const int N){
    size_t element_num_per_thread = (N + SECTION_SIZE - 1) / SECTION_SIZE;
    size_t element_offset = threadIdx.x * element_num_per_thread;
    float* cur_thread_input = input + element_offset;
    float* cur_thread_output = out + element_offset;

    // step1:多线程分段求前缀和
    if(element_offset < N) cur_thread_output[0] = cur_thread_input[0];
    for(int i = 1; i < element_num_per_thread; i++){
        if(element_offset + i < N){
            cur_thread_output[i] = cur_thread_input[i] + cur_thread_output[i - 1];
        }
    }
    __syncthreads(); //确保每个线程计算完前缀和
    // step2:kogge-stone算法扫每一段尾部元素的前缀和
    __shared__ float temp_out[SECTION_SIZE];
    uint32_t global_idx = (threadIdx.x + 1) * element_num_per_thread - 1;
    if(global_idx < N) temp_out[threadIdx.x] = out[global_idx];
    else temp_out[threadIdx.x] = 0.0f;

    for(int stride = 1; stride < SECTION_SIZE; stride <<= 1){
        __syncthreads();
        float val = temp_out[threadIdx.x];
        if (threadIdx.x >= stride) {
            val += temp_out[threadIdx.x - stride];
        }
        __syncthreads();
        temp_out[threadIdx.x] = val;
    }
    __syncthreads();
    
    // step3:将前一段尾部元素的前缀和加到后一段上
    if(threadIdx.x > 0){
        float last_sum = temp_out[threadIdx.x - 1];
        // printf("idx[%d] last_sum:%f\n",threadIdx.x, last_sum);
        for(int i = 0; i < element_num_per_thread; i++){
            if(element_offset + i < N){
                cur_thread_output[i] += last_sum;
            }
        }
    }

}

void thread_coarsening_prefix_sum_host(float* input, float* out, const int N){
    float *d_input, *d_output;

    cudaMalloc((void **)&d_input, N * sizeof(float));
    cudaMalloc((void **)&d_output, N * sizeof(float));

    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0 , N * sizeof(float));

    dim3 step1_block(SET_SECTION_SIZE);
    dim3 step2_grid(1);
    thread_coarsening_prefix_sum<SET_SECTION_SIZE><<<step2_grid, step1_block>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(out, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    // for(int idx = 0; idx < N; idx++){
    //     printf("idx[%d]: %f\n", idx, out[idx]);
    // }
}

template<size_t SECTION_SIZE>
__global__ void brent_kung_prefix_sum(float* input, float* out, const int N, const int element_stride=1){
    __shared__ float temp_out[SECTION_SIZE];
    // 只需要一半的thread，每个thread负责搬运2个元素到共享内存
    uint32_t global_idx = 2 * blockIdx.x * blockDim.x + (threadIdx.x * element_stride);
    if(global_idx < N) temp_out[threadIdx.x] = input[global_idx];
    if(global_idx + blockDim.x < N) temp_out[threadIdx.x + blockDim.x] = input[global_idx + blockDim.x];
    //正向树做一遍reduce
    for(int stride = 1; stride <= blockDim.x; stride <<= 1){
        __syncthreads();
        uint32_t index = (threadIdx.x + 1) * 2 * stride - 1;
        if(index < SECTION_SIZE) temp_out[index] += temp_out[index - stride];
    }
    //反向树分配部分和
    for(int stride = SECTION_SIZE / 4; stride > 0; stride >>= 1){
        __syncthreads();
        uint32_t index = (threadIdx.x + 1) * 2 * stride - 1;
        if(index + stride < SECTION_SIZE) temp_out[index + stride] += temp_out[index];
    }
    __syncthreads();
    //每个线程搬运两个数
    if(global_idx < N) out[global_idx] = temp_out[threadIdx.x];
    if(global_idx + blockDim.x < N) out[global_idx + blockDim.x] = temp_out[threadIdx.x + blockDim.x];


}

void brent_kung_prefix_sum_host(float* input, float* out, const int N){
    float *d_input, *d_output;

    cudaMalloc((void **)&d_input, N * sizeof(float));
    cudaMalloc((void **)&d_output, N * sizeof(float));

    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0 , N * sizeof(float));

    size_t num_blocks = (N + SET_SECTION_SIZE - 1) / SET_SECTION_SIZE;
    dim3 step1_block(SET_SECTION_SIZE / 2);
    dim3 step2_grid(num_blocks);
    brent_kung_prefix_sum<SET_SECTION_SIZE><<<step2_grid, step1_block>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(out, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    float last_block_sum = 0;
    for(int idx = 0; idx < N; idx++){
        if(idx % SET_SECTION_SIZE == 0 && idx != 0){
            last_block_sum = out[idx - 1];
        }
        out[idx] += last_block_sum;
    }
}

int main(){
    const int N = 200000000;
    const size_t size =  N;
    float *input_host = (float*)malloc(size * sizeof(float));
    float *output_host_cpu = (float*)malloc(size * sizeof(float));
    float *output_host_gpu = (float*)malloc(size * sizeof(float));

    for(int i = 0; i < size; i++){
        input_host[i] = 2.0 * (float)drand48() - 1.0;
        // input_host[i] = i;
    }
    {
        Perf perf("cpu_prefix_sum");
        cpu_prefix_sum(input_host, output_host_cpu, N);
    }

    {
        Perf perf("kogge_stone_prefix_sum");
        kogge_stone_prefix_sum_host(input_host, output_host_gpu, N);
    }

    {
        Perf perf("thread_coarsening_prefix_sum");
        thread_coarsening_prefix_sum_host(input_host, output_host_gpu, N);
    }

    {
        Perf perf("brent_kung_prefix_sum");
        brent_kung_prefix_sum_host(input_host, output_host_gpu, N);
    }

    
    if(check_result(output_host_gpu, output_host_cpu, size)){
        printf("test pass\n");
        return 0;
    };

    

    printf("test fail\n");
    return 0;
    
}