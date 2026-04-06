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

// only 1 warp per block(32 threads), m16n16k16. A, B, C: all row_major.
// 每个block负责一个16*16的输出块
// 类似于滑动窗口+线程粗化
template <const int WMMA_M = 16, const int WMMA_N = 16, const int WMMA_K = 16>
__global__ void hgemm_wmma_m16n16k16_naive_kernel(half *A, half *B, half *C,
                                                  int M, int N, int K) {
  const int NUM_K_TILES = div_ceil(K, WMMA_K);
  //先偏移到本block负责的区域
  const int load_gmem_a_m = blockIdx.y * WMMA_M;
  const int load_gmem_b_n = blockIdx.x * WMMA_N;
  if (load_gmem_a_m >= M && load_gmem_b_n >= N)
    return;
  //fragment，是一个warp共享的，warp中线程共同对其做操作
  /*
    template <
        typename Use, 可选值：1、wmma::matrix_a	输入矩阵 A  2、wmma::matrix_b 输入矩阵 B  3、wmma::accumulator	输出矩阵 C（累加器）
        int M,
        int N,
        int K,
        typename T,  可选值：1、half FP16 2、__nv_bfloat16 BF16 3、int8_t INT8
        typename Layout = void
    >
    class fragment;
  */
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> C_frag;
  wmma::fill_fragment(C_frag, 0.0);

  //都声明为行主序，因为在cuda中内存存储是行主序的
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                   wmma::row_major>
        A_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                wmma::row_major>
        B_frag;
  //只tiling K即可，M、N两维度都正好是16
#pragma unroll
  for (int k = 0; k < NUM_K_TILES; ++k) {
    /*
    wmma::load_matrix_sync 用来把内存中的一个矩阵 tile 加载到 fragment（寄存器）里
     参数说明：
     frag 目标 fragment（warp 级寄存器块），可以是：matrix_a / matrix_b / accumulator
     ptr  指向矩阵起始位置的指针，可以是 global memory 或 shared memory
     ldm（leading dimension），行/列之间的跨度（stride）
         如果是 row_major：ldm = 一行有多少元素 如果是 col_major：ldm = 一列有多少元素
     layout（只在 accumulator 时需要）
        指定内存布局：
         wmma::mem_row_major
         wmma::mem_col_major
     ----------------------
     使用特点：
     - warp 级操作，必须 32 个线程一起调用
     - 每个线程只加载一部分数据，最终拼成一个 tile
     - tile 大小由 fragment 的模板参数 M/N/K 决定（比如 16x16x16）
     - 只是加载数据，不做计算
     ----------------------
     注意事项：
     - 所有线程必须执行（不能分支）
     - ldm 必须正确，否则结果错
     - 内存最好对齐（影响性能）
     - matrix_a / matrix_b 的 layout 在 fragment 类型里指定
    */
    wmma::load_matrix_sync(A_frag, A + load_gmem_a_m * K + k * WMMA_K, K);
    wmma::load_matrix_sync(B_frag, B + (k * WMMA_K) * N + load_gmem_b_n, N);

    wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);

    __syncthreads();
  }
  wmma::store_matrix_sync(C + load_gmem_a_m * N + load_gmem_b_n, C_frag, N,
                          wmma::mem_row_major);
}
#define LDST32BITS(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST64BITS(value) (reinterpret_cast<float2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
// m16n16k16 wmma  + tile MMA with smem,  A, B, C: all row_major.
//提升数据复用率(加载数据被多个warp复用，减少重复加载)
//计算密度上升
//warp更多便于延迟隐藏
template <const int WMMA_M = 16, const int WMMA_N = 16, const int WMMA_K = 16,
          const int WMMA_TILE_M = 4, const int WMMA_TILE_N = 2>
__global__ void hgemm_wmma_m16n16k16_mma4x2_kernel(half *A, half *B, half *C,
                                                   int M, int N, int K) {
  // 256 threads(8 warps) per block.
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int NUM_K_TILES = div_ceil(K, WMMA_K);
  //WMMA_TILE_M、WMMA_TILE_N是对应维的warp数
  //每个warp负责WMMA_M、WMMA_N、WMMA_K维度的mma计算
  constexpr int BM = WMMA_M * WMMA_TILE_M;      // 16x4=64，每个block负责的M维数
  constexpr int BN = WMMA_N * WMMA_TILE_N;      // 16x2=32，每个block负责的N维数
  constexpr int BK = WMMA_K;                    // 16
  __shared__ half s_a[BM][BK], s_b[BK][BN]; // 64x16x2=2KB, 16x32x2=1KB

  // 要保证相同的warp下thread执行相同的指令
  // warp_id 0 -> warp_m 0, warp_n 0
  // warp_id 1 -> warp_m 0, warp_n 1
  // warp_id 2 -> warp_m 1, warp_n 0
  // warp_id 3 -> warp_m 1, warp_n 1
  const int tid = threadIdx.y * blockDim.x + threadIdx.x; //tid是block内一维id
  const int warp_id = tid / WARP_SIZE; // 0~7 warp_id within block
  const int lane_id = tid % WARP_SIZE; // 0~31
  const int warp_m = warp_id / WMMA_TILE_N;      // 0,1,2,3
  const int warp_n = warp_id % WMMA_TILE_N;      // 0,1

  // 256线程分别load s_a=64x16, s_b=16x32
  // 64*16/256=4, half4, 16x32/256=2, half2
  const int num_threads = blockDim.x * blockDim.y;
  const int num_elm_per_a = BM * BK / num_threads;
  const int num_elm_per_b = BN * BK / num_threads;
  // s_a, 64*16, 每个线程load 4 half, 每行需要4线程，64行，共256线程
  const int load_smem_a_m = tid / (BK / num_elm_per_a);       // 0~63
  const int load_smem_a_k = (tid % (BK / num_elm_per_a)) * num_elm_per_a; // 0,4,12,...
  // s_b, 16x32, 每个线程load 2 half, 每行需要16线程，16行，共256线程
  const int load_smem_b_k = tid / (BN / num_elm_per_b);                // 0~16
  const int load_smem_b_n = (tid % (BN / num_elm_per_b)) * num_elm_per_b;          // 0,2,4,...,32
  const int load_gmem_a_m = by * BM + load_smem_a_m; // global m，本thread在全局内存的m坐标
  const int load_gmem_b_n = bx * BN + load_smem_b_n; // global n，本thread在全局内存的n坐标

  if (load_gmem_a_m >= M && load_gmem_b_n >= N)
    return;

  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> C_frag;
  wmma::fill_fragment(C_frag, 0.0);
  //fragment是warp级别的
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                   wmma::row_major>
        A_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                   wmma::row_major>
        B_frag;

#pragma unroll
  for (int k = 0; k < NUM_K_TILES; ++k) {
    int load_gmem_a_k = k * WMMA_K + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * WMMA_K + load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
    
    if (num_elm_per_a == 4){
        // 64 bits sync memory issues gmem_a -> smem_a.
        LDST64BITS(s_a[load_smem_a_m][load_smem_a_k]) =
        (LDST64BITS(A[load_gmem_a_addr]));
    }
    else{
        //适配不同warp行列数组合，这里不能使用向量指令取数了
        for(int i = 0; i < num_elm_per_a; ++i){
            s_a[load_smem_a_m][load_smem_a_k + i] = A[load_gmem_a_addr + i];
        }
    }

    if (num_elm_per_b == 2){
            // 32 bits sync memory issues gmem_b -> smem_b.
        LDST32BITS(s_b[load_smem_b_k][load_smem_b_n]) =
        (LDST32BITS(B[load_gmem_b_addr]));
    }
    else{
        for(int i = 0; i < num_elm_per_b; ++i){
            s_b[load_smem_b_k][load_smem_b_n + i] = B[load_gmem_b_addr + i];
        }
    }
    
    __syncthreads();
    //注意共享内存行主序的行stride为BK、BN
    //由于BK等于WMMA_K，故在第K维不需要做偏移
    // warp_m取值0、1、2、3，warp_n取值0、1,8个warp把这块数据都计算完了
    wmma::load_matrix_sync(A_frag, &s_a[warp_m * WMMA_M][0],
                           BK); // BM*BK, BK=WMMA_K
    wmma::load_matrix_sync(B_frag, &s_b[0][warp_n * WMMA_N],
                           BN); // BK*BN, BK=WMMA_K

    wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);

    __syncthreads();
  }

  const int store_gmem_a_m = by * BM + warp_m * WMMA_M;
  const int store_gmem_a_n = bx * BN + warp_n * WMMA_N;
  wmma::store_matrix_sync(C + store_gmem_a_m * N + store_gmem_a_n, C_frag, N,
                          wmma::mem_row_major);
}

// m16n16k16 wmma  + tile MMA with smem,  A, B, C: all row_major.
template <const int WMMA_M = 16, const int WMMA_N = 16, const int WMMA_K = 16,
          const int WMMA_TILE_M = 4, const int WMMA_TILE_N = 2,
          const int WARP_TILE_M = 2, const int WARP_TILE_N = 4> //一个warp负责2*4个16x16x16矩阵乘
__global__ void hgemm_wmma_m16n16k16_mma4x2_warp2x4_kernel(half *A, half *B,
                                                           half *C, int M,
                                                           int N, int K) {
  // 256 threads(8 warps) per block.
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int NUM_K_TILES = div_ceil(K, WMMA_K);
  constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M; // 16x4*2=128
  constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N; // 16x2*4=128
  constexpr int BK = WMMA_K;                             // 16
  __shared__ half s_a[BM][BK], s_b[BK][BN];              // 16x128x2=4KB

  // 要保证相同的warp下thread执行相同的指令
  // warp_id 0 -> warp_m 0, warp_n 0
  // warp_id 1 -> warp_m 0, warp_n 1
  // warp_id 2 -> warp_m 1, warp_n 0
  // warp_id 3 -> warp_m 1, warp_n 1
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int warp_id = tid / WARP_SIZE; // 0~7 warp_id within block
  const int lane_id = tid % WARP_SIZE; // 0~31
  const int warp_m = warp_id / 2;      // 0,1,2,3
  const int warp_n = warp_id % 2;      // 0,1

  // 0. 先计算shared memory中的索引
  // 每个数据16bit
  // tid和需要加载的smem s_a[BM][BK] 之间的索引关系 BM=128 BK=16 按行读取 A行主序
  // 对于s_a每行16个数据，每个线程读取8个half，每行需要2个线程；总共128行，需要128x2刚好256线程
  int load_smem_a_m = tid / 2;                // row 0~127
  int load_smem_a_k = (tid % 2 == 0) ? 0 : 8; // col 0,8  相等判断比乘法快
  // tid和需要加载的smem s_b[BK][BN] 之间的索引关系 BK=16 BN=128 按行读取
  // B行主序
  // 对于s_b每行128个数据，每个线程读8个half，需要16个线程；总共16行，需要16x16=256个线程
  int load_smem_b_k = tid / 16;       // row 0~15
  int load_smem_b_n = (tid % 16) * 8; // col 0,8,...,120
  // 1. 再计算全局内存中的索引
  // 要加载到s_a中的元素对应到A全局内存中的行数
  // 每个block负责出C中大小为BM*BN的块
  int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
  int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c
  if (load_gmem_a_m >= M || load_gmem_b_n >= N)
    return;
  //fragment是warp级别的，因此要创建多个
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                   wmma::row_major>
        A_frag[WARP_TILE_M];
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                   wmma::row_major>
        B_frag[WARP_TILE_N];
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half>
      C_frag[WARP_TILE_M][WARP_TILE_N];

#pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      wmma::fill_fragment(C_frag[i][j], 0.0);
    }
  }

#pragma unroll
  for (int k = 0; k < NUM_K_TILES; ++k) {
    int load_gmem_a_k = k * WMMA_K + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * WMMA_K + load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
    LDST128BITS(s_b[load_smem_b_k][load_smem_b_n]) =
        (LDST128BITS(B[load_gmem_b_addr]));
    LDST128BITS(s_a[load_smem_a_m][load_smem_a_k]) =
        (LDST128BITS(A[load_gmem_a_addr]));
    __syncthreads();


#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      // load 2 tiles -> reg, smem a -> frags a, warp_m 0~3
      // 先定位到自己是warp_m个warp，每个warp负责WMMA_M * WARP_TILE_M行
      // 再定位到是warp内的第几个fragment行
      const int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
      wmma::load_matrix_sync(A_frag[i], &s_a[warp_smem_a_m][0],
                             BK); // BM*BK, BK=WMMA_K
    }

#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      // load 4 tiles -> reg, smem b -> frags b, warp_n 0~2
      // 加载B fragment同理
      const int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
      wmma::load_matrix_sync(B_frag[j], &s_b[0][warp_smem_b_n],
                             BN); // BM*BK, BK=WMMA_K
    }
    //计算矩阵乘，这里C完成了累加
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
      }
    }
    __syncthreads();
  }
  
#pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      const int store_gmem_a_m =
          by * BM + warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
      const int store_gmem_a_n =
          bx * BN + warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
      wmma::store_matrix_sync(C + store_gmem_a_m * N + store_gmem_a_n,
                              C_frag[i][j], N, wmma::mem_row_major);
    }
  }
}

// Double buffers
template <const int WMMA_M = 16, const int WMMA_N = 16, const int WMMA_K = 16,
          const int WMMA_TILE_M = 4, const int WMMA_TILE_N = 2,
          const int WARP_TILE_M = 2, const int WARP_TILE_N = 4,
          const int OFFSET = 0>
__global__ void
//ping pong buffer增加访存和计算的overlap
hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async_kernel(half *A, half *B, half *C,
                                                      int M, int N, int K) {
  // 256 threads(8 warps) per block.
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int NUM_K_TILES = div_ceil(K, WMMA_K);
  constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M; // 16x4*2=128
  constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N; // 16x2*4=128
  constexpr int BK = WMMA_K;                             // 16
  // 16x128x2=4KB, 4+4=8KB, padding to reduce bank conflicts.
  __shared__ half s_a[2][BM][BK + OFFSET], s_b[2][BK][BN + OFFSET];

  // 要保证相同的warp下thread执行相同的指令
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int warp_id = tid / WARP_SIZE; // 0~7 warp_id within block
  const int lane_id = tid % WARP_SIZE; // 0~31
  const int warp_m = warp_id / 2;      // 0,1,2,3
  const int warp_n = warp_id % 2;      // 0,1

  // 0. 先计算shared memory中的索引
  // tid和需要加载的smem s_a[BM][BK] 之间的索引关系 BM=128 BK=8 按行读取 A行主序
  // 对于s_a每行16个数据，每个线程读取8个，需要2个线程；总共128行，需要128x2刚好256线程
  int load_smem_a_m = tid / 2;                // row 0~127
  int load_smem_a_k = (tid % 2 == 0) ? 0 : 8; // col 0,8
  // tid和需要加载的smem s_b[BK][BN] 之间的索引关系 BK=16 BN=128 按行读取
  // B行主序
  // 对于s_b每行128个数据，每个线程读8个数据，需要16个线程；总共16行，需要16x16=256个线程
  int load_smem_b_k = tid / 16;       // row 0~15
  int load_smem_b_n = (tid % 16) * 8; // col 0,8,...,120
  // 1. 再计算全局内存中的索引
  // 要加载到s_a中的元素对应到A全局内存中的行数
  // 每个block负责出C中大小为BM*BN的块
  int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
  int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c
  if (load_gmem_a_m >= M || load_gmem_b_n >= N)
    return;

  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                   wmma::row_major>
        A_frag[WARP_TILE_M];
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                   wmma::row_major>
        B_frag[WARP_TILE_N];
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half>
      C_frag[WARP_TILE_M][WARP_TILE_N];

#pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      wmma::fill_fragment(C_frag[i][j], 0.0);
    }
  }

  // k = 0 is loading here, buffer 0
  //预先填充k为0时的buffer0
  {
    int load_gmem_a_k = load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
    //__cvta_generic_to_shared:把通用地址（generic pointer）转换成 shared memory 地址（用于底层指令）
    uint32_t load_smem_a_ptr =
        __cvta_generic_to_shared(&s_a[0][load_smem_a_m][load_smem_a_k]);
    //异步从 global memory 拷贝到 shared memory（不阻塞），每个线程都执行
    //从Hopper后有专门的TMA引擎做异步拷贝
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16); //读取16B也就是8个Half

    uint32_t load_smem_b_ptr =
        __cvta_generic_to_shared(&s_b[0][load_smem_b_k][load_smem_b_n]);
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
    //提交当前这批 cp.async 请求，让硬件开始执行
    CP_ASYNC_COMMIT_GROUP();
    //等待这批cp.async请求完成
    CP_ASYNC_WAIT_GROUP(0);
  }
  __syncthreads();

#pragma unroll
  for (int k = 1; k < NUM_K_TILES; ++k) { // start from 1
    int smem_sel = (k - 1) & 1;           // k 1->0, k 2->1, k 3->0, ...
    int smem_sel_next = k & 1;            // k 1->1, k 2->0, k 3->1, ...
    //k内流水搬数
    int load_gmem_a_k = k * WMMA_K + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * WMMA_K + load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

    uint32_t load_smem_a_ptr = __cvta_generic_to_shared(
        &s_a[smem_sel_next][load_smem_a_m][load_smem_a_k]);
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

    uint32_t load_smem_b_ptr = __cvta_generic_to_shared(
        &s_b[smem_sel_next][load_smem_b_k][load_smem_b_n]);
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);

    CP_ASYNC_COMMIT_GROUP();

#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      // load 2 tiles -> reg, smem a -> frags a, warp_m 0~3
      const int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
      wmma::load_matrix_sync(A_frag[i], &s_a[smem_sel][warp_smem_a_m][0],
                             BK + OFFSET);
    }

#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      // load 4 tiles -> reg, smem b -> frags b, warp_n 0~2
      const int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
      wmma::load_matrix_sync(B_frag[j], &s_b[smem_sel][0][warp_smem_b_n],
                             BN + OFFSET);
    }

#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
      }
    }

    
    CP_ASYNC_WAIT_GROUP(0);

    __syncthreads();
  }

  // processing last k tile
  {
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                   wmma::row_major>
        A_frag[WARP_TILE_M];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                   wmma::row_major>
        B_frag[WARP_TILE_N];

#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      // load 2 tiles -> reg, smem a -> frags a, warp_m 0~3
      const int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
      wmma::load_matrix_sync(A_frag[i], &s_a[1][warp_smem_a_m][0], BK + OFFSET);
    }

#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      // load 4 tiles -> reg, smem b -> frags b, warp_n 0~2
      const int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
      wmma::load_matrix_sync(B_frag[j], &s_b[1][0][warp_smem_b_n], BN + OFFSET);
    }

#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
      }
    }
  }

// finally, store back to C matrix.
#pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      const int store_gmem_a_m =
          by * BM + warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
      const int store_gmem_a_n =
          bx * BN + warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
      wmma::store_matrix_sync(C + store_gmem_a_m * N + store_gmem_a_n,
                              C_frag[i][j], N, wmma::mem_row_major);
    }
  }
}


void hgemm_wmma_m16n16k16_mma4x2(half *A, half *B, half *C, size_t M, size_t N, size_t K) {
  
  constexpr int WMMA_M = 16;
  constexpr int WMMA_N = 16;
  constexpr int WMMA_K = 16;
  constexpr int WMMA_TILE_M = 4;
  constexpr int WMMA_TILE_N = 2;
  constexpr int NUM_THREADS =
      (WMMA_TILE_M * WMMA_TILE_N * WARP_SIZE); // 4 * 2 * 32 = 256

  dim3 block(NUM_THREADS);
  dim3 grid(div_ceil(N, WMMA_N * WMMA_TILE_N),
            div_ceil(M, WMMA_M * WMMA_TILE_M));

  hgemm_wmma_m16n16k16_mma4x2_kernel<WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M,
                                     WMMA_TILE_N>
      <<<grid, block>>>(A, B, C, M, N, K);
}

void hgemm_wmma_m16n16k16_mma4x2_warp2x4(half *A, half *B, half *C, size_t M, size_t N, size_t K) {
  
  constexpr int WMMA_M = 16;
  constexpr int WMMA_N = 16;
  constexpr int WMMA_K = 16;
  constexpr int WMMA_TILE_M = 4;
  constexpr int WMMA_TILE_N = 2;
  constexpr int WARP_TILE_M = 2;
  constexpr int WARP_TILE_N = 4;
  constexpr int NUM_THREADS =
      (WMMA_TILE_M * WMMA_TILE_N * WARP_SIZE); // 4 * 2 * 32 = 256

  dim3 block(NUM_THREADS);
  dim3 grid(div_ceil(N, WMMA_N * WMMA_TILE_N * WARP_TILE_N),
            div_ceil(M, WMMA_M * WMMA_TILE_M * WARP_TILE_M));

  hgemm_wmma_m16n16k16_mma4x2_warp2x4_kernel<WMMA_M, WMMA_N, WMMA_K,
                                             WMMA_TILE_M, WMMA_TILE_N,
                                             WARP_TILE_M, WARP_TILE_N>
      <<<grid, block>>>(A, B, C, M, N, K);
}

// double buffer, padding
void hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async(half *A, half *B, half *C, size_t M, size_t N, size_t K) {
  constexpr int WMMA_M = 16;
  constexpr int WMMA_N = 16;
  constexpr int WMMA_K = 16;
  constexpr int WMMA_TILE_M = 4;
  constexpr int WMMA_TILE_N = 2;
  constexpr int WARP_TILE_M = 2;
  constexpr int WARP_TILE_N = 4;
  constexpr int NUM_THREADS =
      (WMMA_TILE_M * WMMA_TILE_N * WARP_SIZE); // 4 * 2 * 32 = 256

  dim3 block(NUM_THREADS);
  dim3 grid(div_ceil(N, WMMA_N * WMMA_TILE_N * WARP_TILE_N),
            div_ceil(M, WMMA_M * WMMA_TILE_M * WARP_TILE_M));

  hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async_kernel<
      WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N, WARP_TILE_M,
      WARP_TILE_N, 8><<<grid, block>>>(A, B, C, M, N, K);
}

#define KERNEL_WRAPER_NAME(kernel) kernel##_wraper

#define KERNEL_WRAPER_DEFINE(kernel) void KERNEL_WRAPER_NAME(kernel)(half *A, half *B, half *C, size_t M, size_t N, size_t K) \
{    \
    constexpr int WMMA_M = 16;  \
    constexpr int WMMA_N = 16;  \
    constexpr int WMMA_K = 16;  \
    dim3 block(WARP_SIZE);      \
    dim3 grid(div_ceil(N, WMMA_N), div_ceil(M, WMMA_M));  \
    kernel<WMMA_M, WMMA_N, WMMA_K>      \
        <<<grid, block>>>(A, B, C, M, N, K);  \
}

KERNEL_WRAPER_DEFINE(hgemm_wmma_m16n16k16_naive_kernel)

int main(){
    Tester tester(512, 1024, 1024, 1, 10, 100,
                  true);
    tester.evaluate(KERNEL_WRAPER_NAME(hgemm_wmma_m16n16k16_naive_kernel), "hgemm_wmma_m16n16k16_naive");
    tester.evaluate(hgemm_wmma_m16n16k16_mma4x2, "hgemm_wmma_m16n16k16_mma4x2");
    tester.evaluate(hgemm_wmma_m16n16k16_mma4x2_warp2x4, "hgemm_wmma_m16n16k16_mma4x2_warp2x4");
    tester.evaluate(hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async, "hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async");
    
}