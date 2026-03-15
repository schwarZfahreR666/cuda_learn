#include <cstddef>
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

static int cmp_int(const void* a, const void* b){
    return *(const int*)a - *(const int*)b;
}

static bool is_sorted_non_descending(const int* arr, size_t size){
    size_t zero_num = 0;
    if(arr[0] == 0) zero_num++;
    for(size_t i = 1; i < size; i++){
        if(arr[i] < arr[i - 1]){
            printf("idx is [%ld], a[idx-1]=%d, a[idx]=%d\n", i, arr[i-1], arr[i]);
            return false;
        } 
        if(arr[i] == 0) zero_num++;
    }
    return !(zero_num == size);
}

void merge_cpu(int* A, int m, int* B, int n, int* C){
    int i = 0;
    int j = 0;
    int k = 0;
    while(i < m && j < n){
        if(A[i] <= B[j]){
            C[k++] = A[i++];
        }
        else{
            C[k++] = B[j++];
        }
    }

    while(i < m) C[k++] = A[i++];
    while(j < n) C[k++] = B[j++];
}

// 共享内存 padding：每 32 个逻辑元素多 1 个物理位置，避免 stride-32 的 bank 冲突
__device__ __forceinline__ int pad_idx(int i) { return i + (i >> 5); }

//给定C中的元素k，找到i和j(k-i)，使得A[i-1]<=B[j]并且B[j-1]<A[i]
//子数组C[0]~C[k-1](k个数)来自于A[0]~A[i-1](i个数) 和 B[0]~B[j-1](j个数)
//k = i + j
__device__ int co_rank(int k, int* A, int m, int* B, int n){
    int i = k < m ? k : m; // i = min(k, m) i是A数组的索引，不能大于m，也不能大于k
    int j = k - i; //j是i的共秩值

    int i_low = k < n ? 0 : k - n; //若k > n，那么i最少也有k-n
    int j_low = k < m ? 0 : k - m; //与上同理
    
    int delta;
    bool active = true;
    while(active){
        //若C[k-1]来自于A，那么B[j] >= A[i-1]
        //此时因为 B[j-1]一定在A[i-1]前面被选，故B[j-1] < A[i-1] <= A[i]成立
        if(i > 0 && j < n && A[i - 1] > B[j]){ //这种情况i大了，j要向后调整
            delta = (i - i_low + 1) / 2; // ceil(i - i_low) / 2
            j_low = j;
            j = j + delta;
            i = i - delta;
        }
        //若C[k-1]来自于B，那么A[i] > B[j - 1](不能取等，否则会取A[i]不会取B[j-1])
        //此时A[i-1] <= B[j-1] <= B[j]成立
        else if(j > 0 && i < m && B[j - 1] >= A[i]){
            delta = (j - j_low + 1) / 2;
            i_low = i;
            i = i + delta;
            j = j - delta;
        }
        //两项同时满足，完成搜索
        else{
            active = false;
        }
    }

    return i;
}

// 用于 padding 后的 shared 数组：按 pad_idx 索引，消除 bank 冲突
__device__ int co_rank_padded(int k, int* A, int m, int* B, int n){
    int i = k < m ? k : m;
    int j = k - i;
    int i_low = k < n ? 0 : k - n;
    int j_low = k < m ? 0 : k - m;
    int delta;
    bool active = true;
    while(active){
        if(i > 0 && j < n && A[pad_idx(i - 1)] > B[pad_idx(j)]){
            delta = (i - i_low + 1) / 2;
            j_low = j;
            j = j + delta;
            i = i - delta;
        }
        else if(j > 0 && i < m && B[pad_idx(j - 1)] >= A[pad_idx(i)]){
            delta = (j - j_low + 1) / 2;
            i_low = i;
            i = i + delta;
            j = j - delta;
        }
        else{
            active = false;
        }
    }
    return i;
}

__device__ void merge_sequential(int* A, int m, int* B, int n, int* C){
    int i = 0;
    int j = 0;
    int k = 0;
    while(i < m && j < n){
        if(A[i] <= B[j]){
            C[k++] = A[i++];
        }
        else{
            C[k++] = B[j++];
        }
    }

    while(i < m) C[k++] = A[i++];
    while(j < n) C[k++] = B[j++];
}

// 从 padding 后的 shared 数组归并到 global C；A/B 为 shared 基址，a_curr/b_curr 为逻辑起始下标
__device__ void merge_sequential_padded(int* A, int a_curr, int a_len, int* B, int b_curr, int b_len, int* C){
    int i = 0;
    int j = 0;
    int k = 0;
    while(i < a_len && j < b_len){
        if(A[pad_idx(a_curr + i)] <= B[pad_idx(b_curr + j)]){
            C[k++] = A[pad_idx(a_curr + i++)];
        }
        else{
            C[k++] = B[pad_idx(b_curr + j++)];
        }
    }
    while(i < a_len) C[k++] = A[pad_idx(a_curr + i++)];
    while(j < b_len) C[k++] = B[pad_idx(b_curr + j++)];
}

// __launch_bounds__ 限制每 block 线程数，便于编译器降低寄存器占用、提升 occupancy
template<int TILE_SIZE>
__global__ __launch_bounds__(1024) void merge_kernel_basic(int* A, int m, int* B, int n, int* C){
    const int PADDED_SIZE = TILE_SIZE + (TILE_SIZE + 31) / 32;
    __shared__ int A_S[PADDED_SIZE];
    __shared__ int B_S[PADDED_SIZE];
    // step1: block级别tiling
    int num_elements_per_block = (m + n + gridDim.x - 1) / gridDim.x; // 计算每个块负责归并元素数
    int global_C_cur = blockIdx.x * num_elements_per_block; //计算当前块负责C的索引
    int global_C_next = min(global_C_cur + num_elements_per_block, m + n); //计算下一个块负责C的索引

    //计算block级别的tiling数据
    if(threadIdx.x == 0){
        A_S[0] = co_rank(global_C_cur, A, m, B, n); 
        A_S[1] = co_rank(global_C_next, A, m, B, n);
    }

    __syncthreads();

    int A_curr = A_S[0];
    int A_next = A_S[1];
    int B_curr = global_C_cur - A_curr;
    int B_next = global_C_next - A_next;
    __syncthreads();

    
    int counter = 0;
    int C_length = global_C_next - global_C_cur;
    int A_length = A_next - A_curr;
    int B_length = B_next - B_curr;

    int total_iter = (C_length + TILE_SIZE - 1) / TILE_SIZE;

    int C_completed = 0;
    int A_consumed = 0;
    int B_consumed = 0;

    while(counter < total_iter){
        int a_tile_len = min(TILE_SIZE, A_length - A_consumed);
        int b_tile_len = min(TILE_SIZE, B_length - B_consumed);
        int written = min(TILE_SIZE, C_length - C_completed);
        if(written <= 0) break;

        //step2: 将A和B部分载入共享内存
        for(int i = threadIdx.x; i < TILE_SIZE; i += blockDim.x){
            if(i < a_tile_len) A_S[pad_idx(i)] = A[A_curr + A_consumed + i];
        }
        for(int i = threadIdx.x; i < TILE_SIZE; i += blockDim.x){
            if(i < b_tile_len) B_S[pad_idx(i)] = B[B_curr + B_consumed + i];
        }
        __syncthreads();

        // step3: 每个thread完成一部分数据的归并（在本次 tile 内按 written 划分）
        size_t num_element_per_thread = (TILE_SIZE + blockDim.x - 1) / blockDim.x;
        int c_curr = (int)(threadIdx.x * num_element_per_thread);
        int c_next = (int)min((size_t)(c_curr + num_element_per_thread), (size_t)written);
        c_curr = min(c_curr, written);
        int a_curr = co_rank_padded(c_curr, A_S, a_tile_len, B_S, b_tile_len);
        int a_next = co_rank_padded(c_next, A_S, a_tile_len, B_S, b_tile_len);
        int b_curr = c_curr - a_curr;
        int b_next = c_next - a_next;

        merge_sequential_padded(A_S, a_curr, a_next - a_curr, B_S, b_curr, b_next - b_curr,
                               C + global_C_cur + C_completed + c_curr); //C这里使用的是全局内存
        counter++;
        C_completed += written;
        A_consumed += co_rank_padded(written, A_S, a_tile_len, B_S, b_tile_len);
        B_consumed = C_completed - A_consumed;
        __syncthreads();
    }


}

#define SET_TILE_SIZE (4096)
void merge_basic_host(int* A, int m, int* B, int n, int* C){
    int *d_A, *d_B, *d_C;

    cudaMalloc((void **)&d_A, m * sizeof(int));
    cudaMalloc((void **)&d_B, n * sizeof(int));
    cudaMalloc((void **)&d_C, (m + n) * sizeof(int));

    cudaMemcpy(d_A, A, m * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, (m + n) * sizeof(int));

    dim3 step1_block(1024);
    dim3 step1_grid(128);
    merge_kernel_basic<SET_TILE_SIZE><<<step1_grid, step1_block>>>(d_A, m, d_B, n, d_C);
    cudaError_t err = cudaGetLastError();
    printf("%s\n", cudaGetErrorString(err));
    cudaDeviceSynchronize();
    
    cudaMemcpy(C, d_C, (m + n) * sizeof(int), cudaMemcpyDeviceToHost);
}

// 可配置：A 的元素个数、B 的元素个数
#define CONFIG_M 10000000
#define CONFIG_N 10000000

int main(){
    const int m = CONFIG_M;
    const int n = CONFIG_N;
    const size_t total = (size_t)m + n;

    int* A = (int*)malloc((size_t)m * sizeof(int));
    int* B = (int*)malloc((size_t)n * sizeof(int));
    int* C = (int*)malloc(total * sizeof(int));
    if(!A || !B || !C){
        printf("malloc failed (m=%d, n=%d)\n", m, n);
        return 1;
    }

    for(int i = 0; i < m; i++) A[i] = (int)(drand48() * 1000000);
    for(int i = 0; i < n; i++) B[i] = (int)(drand48() * 1000000);
    qsort(A, (size_t)m, sizeof(int), cmp_int);
    qsort(B, (size_t)n, sizeof(int), cmp_int);
    memset(C, 0, total * sizeof(int));
    {
        Perf perf("merge_cpu");
        merge_cpu(A, m, B, n, C);
    }

    bool sorted = is_sorted_non_descending(C, total);
    memset(C, 0, total * sizeof(int));
    {
        Perf perf("merge_basic_host");
        
        merge_basic_host(A, m, B, n, C);
    }
    sorted &= is_sorted_non_descending(C, total);

    free(A);
    free(B);
    free(C);

    if(!sorted){
        printf("merge test fail: C not sorted (m=%d, n=%d)\n", m, n);
        return 1;
    }
    printf("merge test pass (m=%d, n=%d)\n", m, n);
    return 0;
}