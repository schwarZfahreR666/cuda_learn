#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "common/tester.h"
using namespace nvcuda;

#define HOST_DEVICE_INLINE __device__ __host__ inline
#define WARP_SIZE 32
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
//使用wmma+共享内存只计算一个16x16x16的矩阵乘
__global__ void shared_mem_wmma_kernel(half *A, half *B, half *C, int M, int N, int K) {
    __shared__ half smem_a[16 * 16];
    __shared__ half smem_b[16 * 16];
    __shared__ half smem_c[16 * 16];
    int tid = threadIdx.x;

    LDST128BITS(*(smem_a + 8 * tid)) =
        (LDST128BITS(*(A + 8 * tid)));
    LDST128BITS(*(smem_b + 8 * tid)) =
        (LDST128BITS(*(B + 8 * tid)));
    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, half> C_frag;
    wmma::fill_fragment(C_frag, 0.0);

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half,
                    wmma::row_major>
            A_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half,
                    wmma::row_major>
            B_frag;

    wmma::load_matrix_sync(A_frag, smem_a, 16);
    wmma::load_matrix_sync(B_frag, smem_b, 16);

    wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);

    wmma::store_matrix_sync(smem_c, C_frag, 16, wmma::mem_row_major);

    __syncthreads();
    (LDST128BITS(*(C + 8 * tid))) = 
        LDST128BITS(*(smem_c + 8 * tid));
}
#define REG(val) (*reinterpret_cast<uint32_t*>(&(val)))
__device__ __forceinline__ void ldmatrix_sync(half* dst, void* addr){
    asm volatile(                                                                
      "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, "      
      "[%4];\n"                                                                
      : "=r"(REG(dst[0])), "=r"(REG(dst[2])), "=r"(REG(dst[4])), "=r"(REG(dst[6]))                                 
      : "l"(__cvta_generic_to_shared(addr)));
}

__device__ __forceinline__ void ldmatrix_trans_sync(half* dst, void* addr){
    asm volatile(                                                                
      "ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, "      
      "[%4];\n"                                                                
      : "=r"(REG(dst[0])), "=r"(REG(dst[2])), "=r"(REG(dst[4])), "=r"(REG(dst[6]))                                 
      : "l"(__cvta_generic_to_shared(addr)));
}
__device__ __forceinline__ void mma_sync_m16n8k16(half* A, half* B, half* C){
    asm volatile(                                                                
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, "  
      "%4, %5}, {%6, %7}, {%8, %9};\n"                                         
      : "=r"(REG(C[0])), "=r"(REG(C[2]))                                                   
      : "r"(REG(A[0])), "r"(REG(A[2])), "r"(REG(A[4])), "r"(REG(A[6])), 
      "r"(REG(B[0])), "r"(REG(B[2])), 
      "r"(REG(C[0])), "r"(REG(C[2])));
}
  

//使用mma+共享内存只计算一个16x16x16的矩阵乘
__global__ void shared_mem_mma_kernel(half *A, half *B, half *C, int M, int N, int K) {
    __shared__ half smem_a[16 * 16];
    __shared__ half smem_b[16 * 16];
    __shared__ half smem_c[16 * 16];
    int tid = threadIdx.x;

    LDST128BITS(*(smem_a + 8 * tid)) =
        (LDST128BITS(*(A + 8 * tid)));
    LDST128BITS(*(smem_b + 8 * tid)) =
        (LDST128BITS(*(B + 8 * tid)));

    //对 accumulator<16,16,16,half>，C_frag.x 是一个 half[8] 数组（每个线程持有 8 个 FP16 元素）
    //每个线程只看到自己的那一份：C_frag.x[i] 就是 当前线程负责的第 i 个累加器元素
    //所以每个线程的C_frag.x[0]和C_frag.x[1]就对应R0小矩阵
    //C_frag.x[2]和C_frag.x[3]就对应R1小矩阵
    //C_frag.x + 4就偏到了R2、R3那部分
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> C_frag;
    wmma::fill_fragment(C_frag, 0.0);

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half,
                    wmma::row_major>
            A_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half,
                    wmma::row_major>
            B_frag;
    // ldmatrix指令需要32个线程提供不同位置的地址。shared mem分为16x16，每个线程提供一行连续8个数的地址
    // 分为16行，2列
    // t0-t15提供0-15行，第一列的地址；t16-t31提供0-15行，第二列的地址
    uint32_t row = tid % 16;
    uint32_t col = tid / 16;
    ldmatrix_sync(A_frag.x, smem_a + row * 16 + col * 8);
    ldmatrix_trans_sync(B_frag.x, smem_b + row * 16 + col * 8);

    //使用m16n8k16计算m16n16k16的矩阵乘，等于A矩阵不变，B矩阵列切，最后结果做一次拼接即可
    mma_sync_m16n8k16(A_frag.x, B_frag.x, C_frag.x);
    mma_sync_m16n8k16(A_frag.x, B_frag.x + 4, C_frag.x + 4);

    wmma::store_matrix_sync(smem_c, C_frag, 16, wmma::mem_row_major);

    __syncthreads();
    (LDST128BITS(*(C + 8 * tid))) = 
        LDST128BITS(*(smem_c + 8 * tid));
}

template <uint32_t S, uint32_t B, uint32_t M>
__device__ __forceinline__ uint32_t swizzle(uint32_t addr){
    constexpr auto b_mask = ((1 << B) - 1) << M;
    return ((addr >> S) & b_mask) ^ addr;
}

//使用mma+共享内存+swizzle只计算一个16x16x16的矩阵乘
__global__ void shared_mem_mma_swizzle_kernel(half *A, half *B, half *C, int M, int N, int K) {
    __shared__ half smem_a[16 * 16];
    __shared__ half smem_b[16 * 16];
    __shared__ half smem_c[16 * 16];
    int tid = threadIdx.x;

    uint32_t g_addr = tid * 8;
    auto g2s_addr = swizzle<3, 1, 3>(g_addr);

    LDST128BITS(*(smem_a + g2s_addr)) =
        (LDST128BITS(*(A + g_addr)));
    LDST128BITS(*(smem_b + g2s_addr)) =
        (LDST128BITS(*(B + g_addr)));

    //对 accumulator<16,16,16,half>，C_frag.x 是一个 half[8] 数组（每个线程持有 8 个 FP16 元素）
    //每个线程只看到自己的那一份：C_frag.x[i] 就是 当前线程负责的第 i 个累加器元素
    //所以每个线程的C_frag.x[0]和C_frag.x[1]就对应R0小矩阵
    //C_frag.x[2]和C_frag.x[3]就对应R1小矩阵
    //C_frag.x + 4就偏到了R2、R3那部分
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> C_frag;
    wmma::fill_fragment(C_frag, 0.0);

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half,
                    wmma::row_major>
            A_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half,
                    wmma::row_major>
            B_frag;
    // ldmatrix指令需要32个线程提供不同位置的地址。shared mem分为16x16，每个线程提供一行连续8个数的地址
    // 分为16行，2列
    // t0-t15提供0-15行，第一列的地址；t16-t31提供0-15行，第二列的地址
    uint32_t r_addr = (tid % 16) * 16 + (tid / 16) * 8;
    auto r2s_addr = swizzle<3, 1, 3>(r_addr);
    ldmatrix_sync(A_frag.x, smem_a + r2s_addr);
    ldmatrix_trans_sync(B_frag.x, smem_b + r2s_addr);

    //使用m16n8k16计算m16n16k16的矩阵乘，等于A矩阵不变，B矩阵列切，最后结果做一次拼接即可
    mma_sync_m16n8k16(A_frag.x, B_frag.x, C_frag.x);
    mma_sync_m16n8k16(A_frag.x, B_frag.x + 4, C_frag.x + 4);

    wmma::store_matrix_sync(smem_c, C_frag, 16, wmma::mem_row_major);

    __syncthreads();
    (LDST128BITS(*(C + 8 * tid))) = 
        LDST128BITS(*(smem_c + 8 * tid));
}

void shared_mem_wmma(half *A, half *B, half *C, size_t M, size_t N, size_t K)
{    
    constexpr int WMMA_M = 16;  
    constexpr int WMMA_N = 16;  
    constexpr int WMMA_K = 16;  
    dim3 block(WARP_SIZE);      
    dim3 grid(div_ceil(N, WMMA_N), div_ceil(M, WMMA_M));  
    shared_mem_wmma_kernel<<<grid, block>>>(A, B, C, M, N, K);  
}

void shared_mem_mma(half *A, half *B, half *C, size_t M, size_t N, size_t K)
{    
    constexpr int WMMA_M = 16;  
    constexpr int WMMA_N = 16;  
    constexpr int WMMA_K = 16;  
    dim3 block(WARP_SIZE);      
    dim3 grid(div_ceil(N, WMMA_N), div_ceil(M, WMMA_M));  
    shared_mem_mma_kernel<<<grid, block>>>(A, B, C, M, N, K);  
}

void shared_mem_mma_swizzle(half *A, half *B, half *C, size_t M, size_t N, size_t K)
{    
    constexpr int WMMA_M = 16;  
    constexpr int WMMA_N = 16;  
    constexpr int WMMA_K = 16;  
    dim3 block(WARP_SIZE);      
    dim3 grid(div_ceil(N, WMMA_N), div_ceil(M, WMMA_M));  
    shared_mem_mma_swizzle_kernel<<<grid, block>>>(A, B, C, M, N, K);  
}

int main(){
    Tester tester(16, 16, 16, 1, 10, 100,
                  true);
    tester.evaluate(shared_mem_wmma, "shared_mem_wmma");
    tester.evaluate(shared_mem_mma, "shared_mem_mma");
    tester.evaluate(shared_mem_mma_swizzle, "shared_mem_mma_swizzle");
    
}