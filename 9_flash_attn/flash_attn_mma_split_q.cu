#include "utils.h"

// FlashAttention-2（Tensor Core MMA / PTX）。Q,K,V,O: [B,H,N,d]。
// 论文: https://arxiv.org/pdf/2307.08691
// 每 block：Q_tile [Br,d]；K,V 沿序列整块扫 [N,d]。
// split-Q：多 warp 分摊 Q 行块，共享读 K/V，减少 warp 间通信。

// MMA = m16n8k16, Br=16x4=64, Bc=8x8=64, layout: 4 warps
// |   64x64      M row |      warp_KV 0       | 这里不再拆分4个warp各负责一段Bc
// | warp_QP 0    0-15  | MMA 0 ... MMA 0 (x8) | 每个warp都要负责全部Bc做8次MMA
// | warp_QP 1    16-31 | MMA 1 ... MMA 1 (x8) | 按kWarpTileSeqLenK循环加载到寄存器中
// | warp_QP 2    32-47 | MMA 2 ... MMA 2 (x8) |
// | warp_QP 3    48-63 | MMA 3 ... MMA 3 (x8) |

// MMA = m16n8k16, Br=16x8=128, Bc=8x16=128, layout: 8 warps
// |  128x128  |      warp_KV 0        |
// | warp_QP 0 | MMA 0 ... MMA 0 (x16) |
// | warp_QP 1 | MMA 1 ... MMA 1 (x16) |
// | warp_QP 2 | MMA 2 ... MMA 2 (x16) |
// | warp_QP 3 | MMA 3 ... MMA 3 (x16) |
// | warp_QP 4 | MMA 4 ... MMA 4 (x16) |
// | warp_QP 5 | MMA 5 ... MMA 5 (x16) |
// | warp_QP 6 | MMA 6 ... MMA 6 (x16) |
// | warp_QP 7 | MMA 7 ... MMA 7 (x16) |

// ---- kernel 模板参数（与 launch 里传入的 constexpr 一致）----
// 分块公式（块内）：
//   Br（Q 行 tile 高度）= kMmaAtomM * kMmaTileSeqLenQ * kWarpTileSeqLenQ
//   Bc（K/V 序列方向 tile 宽度）= kMmaAtomN * kMmaTileSeqLenK * kWarpTileSeqLenK
//   每 block 线程数 = 32 * kMmaTileSeqLenQ * kMmaTileSeqLenK（warp 网格为 Q 维 × K 维）
// split-Q：warp_id → warp_QP；沿 Br 的 warp 个数 = kMmaTileSeqLenQ（本文件强制 kMmaTileSeqLenK==1）。
// 单个 warp 内再沿 Br 叠几段 16 行 MMA 由 kWarpTileSeqLenQ 决定（内层循环 i）。

template <
    // kHeadDim：每个 attention head 的维度 d（如 32/64/96/128）。
    // 用于 softmax 的 scale、全局内存步长，以及 P@V 时沿 d 切分的寄存器块数（配合 kWarpTileHeadDimV）。
    const int kHeadDim,

    // kMmaAtomM / kMmaAtomN / kMmaAtomK：一条 Tensor Core MMA（HMMA16816）在逻辑上的 m×n×k。
    // 本工程固定为 16×8×16，与 static_assert 一致；决定单次 MMA 在矩阵里占的「原子格子」大小。
    const int kMmaAtomM,
    const int kMmaAtomN,
    const int kMmaAtomK,

    // kMmaTileSeqLenQ：Q@K^T 阶段，沿 Br（对应 GEMM 的 M 维、Q 的行块）方向有多少个「warp 条带」。
    // 每个条带在一个 warp 里负责，warp_QP ∈ [0, kMmaTileSeqLenQ)（当 kMmaTileSeqLenK==1）。
    // 与 kWarpTileSeqLenQ 相乘后得到 Br 上总共多少个「16 行」的 MMA 原子：Br = 16 * kMmaTileSeqLenQ * kWarpTileSeqLenQ。
    const int kMmaTileSeqLenQ,

    // kMmaTileSeqLenK：Q@K^T 阶段，沿 Bc（对应 GEMM 的 N 维、K 的序列块宽度）方向的 warp 网格第二维长度。
    // 本 kernel 强制为 1（static_assert）；Bc 方向主要靠 kWarpTileSeqLenK 拉长（每个原子 8 列）。
    const int kMmaTileSeqLenK,

    // kMmaTileSeqLenP：P@V 阶段沿 Br（M 维）的 warp 条带数，语义同 Q 侧的 kMmaTileSeqLenQ，通常取相同值。
    const int kMmaTileSeqLenP,

    // kMmaTileHeadDimV：P@V 阶段，沿输出 head 维 d（GEMM 的 N 维）每个 warp 条带内再分几份 8 列 MMA。
    // launch 里固定为 1，即每个 warp 在 d 上先只占一条 8 列原子，整条 d 由 kWarpTileHeadDimV 个这样的块拼满。
    const int kMmaTileHeadDimV,

    // kWarpTileSeqLenQ：单个 warp 内，沿 Br 方向连续堆叠几个「kMmaAtomM 行」的 MMA（内层 for i）。
    // 本 launch 固定为 1；若 >1，则同一 warp_QP 负责 Br 上多段 16 行。
    const int kWarpTileSeqLenQ,

    // kWarpTileSeqLenK：沿 Bc 方向，每个 (kMmaTileSeqLenK) 单元再乘上的倍数；Bc = 8 * kMmaTileSeqLenK * kWarpTileSeqLenK。
    // d<128 时常为 8（Bc=64），d>=128 时为 2（仍配合其它参数得到目标 Bc）。
    const int kWarpTileSeqLenK,

    // kWarpTileSeqLenP：P@V 沿 Br 的 warp 内堆叠因子，与 kWarpTileSeqLenQ 对应；本 launch 固定为 1。
    const int kWarpTileSeqLenP,

    // kWarpTileHeadDimV：沿 d 维共有多少个「kMmaAtomN * kMmaTileHeadDimV」宽的子块（每块 8 列当 kMmaTileHeadDimV==1）。
    // 须满足 kHeadDim == kMmaAtomN * kMmaTileHeadDimV * kWarpTileHeadDimV（见 static_assert）。
    const int kWarpTileHeadDimV,

    // kStage：K（及流水相关的 V）在共享内存中的缓冲「段」数。1 = 无多缓冲、每步同步加载；2 = 双缓冲 + cp.async 与当前块计算重叠。
    const int kStage,

    // kPad：共享内存里每一行在 d 维后的额外 half 元素数，用于对齐与缓解 bank 冲突；须为非负且为 8 的倍数。
    const int kPad>
__global__ void __launch_bounds__(WARP_SIZE *kMmaTileSeqLenQ *kMmaTileSeqLenK)
    flash_attn_mma_stages_split_q_kernel(half *Q, half *K, half *V, half *O,
                                         int QKV_seqlen, int QKV_head) {
  // QK^T: NT；PV: NN。K 存 [Bc,d] 行主序，视作 K^T 的列主视角。
  static_assert(kMmaAtomM == 16 && kMmaAtomN == 8 && kMmaAtomK == 16);
  static_assert(kMmaTileSeqLenQ <= 8 && kMmaTileSeqLenK == 1); // Q@K^T：Bc 向仅 1 列 warp
  static_assert(kMmaTileSeqLenP <= 8 && kMmaTileHeadDimV == 1);
  static_assert(kWarpTileSeqLenQ == 1 && kWarpTileSeqLenK <= 16);
  static_assert(kWarpTileSeqLenP == 1 &&
                kWarpTileHeadDimV ==
                    (kHeadDim / (kMmaAtomN * kMmaTileHeadDimV)));
  static_assert(kStage > 0 && kStage < 3);
  static_assert(kPad >= 0 && kPad % 8 == 0);
  // Br/Bc：Q 行块与 K/V 序列块；kNumThreads = warp 数 × 32
  // MMA的m数 * 沿Br方向分多少个warp * 每个warp中分多少个MMA，Bc同理
  constexpr int Br =
      kMmaAtomM * kMmaTileSeqLenQ * kWarpTileSeqLenQ; // 16*4*1=64
  constexpr int Bc =
      kMmaAtomN * kMmaTileSeqLenK * kWarpTileSeqLenK; //  8*1*8=64
  constexpr int kNumThreads =
      WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK; // 32*4*1=128, num threads
  // Now, N must be mutliples of Bc(32/64) for KV tiling across seqlen.
  const int Tc = div_ceil(QKV_seqlen, Bc); // Tc K_tile. d / Bc
  const float scale = 1.0f / sqrt((float)kHeadDim);

  // grid.x: 按Q 行 tile；grid.y: batch×head
  const int QKV_batch_id = blockIdx.y / QKV_head; // Batch size
  const int QKV_head_id = blockIdx.y % QKV_head;  // Head num
  const int Q_tile_id = blockIdx.x;               // Q tile_id, range [0, Tr]
  const int O_tile_id = Q_tile_id;                // O tile_id, same as Q.
  const int tid = threadIdx.x;                    // within block
  const int warp_id = tid / WARP_SIZE;            // 0~7 warp_id within block
  const int lane_id = tid % WARP_SIZE;            // 0~31
  const int warp_QP = warp_id;                    // 0,1,2,3 or 0~7
  const int warp_KV = 0;                          // 0
  // [B,H,N,d] 偏移到对应的 [N, d]
  const int Q_gmem_offset =
      ((QKV_batch_id * QKV_head * QKV_seqlen * kHeadDim) +
       (QKV_head_id * QKV_seqlen * kHeadDim)); // Q [seqlen,d]
  const int K_gmem_offset =
      ((QKV_batch_id * QKV_head * QKV_seqlen * kHeadDim) +
       (QKV_head_id * QKV_seqlen * kHeadDim)); // K [seqlen,d]
  const int V_gmem_offset = Q_gmem_offset;     // V [seqlen,d]
  const int O_gmem_offset = Q_gmem_offset;     // O [seqlen,d]

  // tid -> smem 内偏移（协作加载 Q_tile / K,V tile）
  // kNumThreads / Br代表几个thread负责一个Q tile的一行
  int load_smem_Q_Br = (tid / (kNumThreads / Br)); // 代表当前thread负责Q tile的哪一行
  int load_smem_Q_d =
      (tid % (kNumThreads / Br)) *
      (kHeadDim / (kNumThreads / Br)); // 当前thread在Q tile一行的偏移量
  // Mapping K gmem -> tid -> smem, K[Bc,d]=[64 or 128,64], 128 threads.
  int load_smem_K_Bc = (tid / (kNumThreads / Bc)); // Bc 64, tid / 2, row 0~64
  int load_smem_K_d =
      (tid % (kNumThreads / Bc)) *
      (kHeadDim / (kNumThreads / Bc)); // (tid % 2) * 32, 0,32,...
  // Mapping V gmem -> tid -> smem, V[Bc,d]=[64,64 or 128], 128 threads.
  int load_smem_V_Bc = (tid / (kNumThreads / Bc)); // Bc 64, tid / 2, row 0~64
  int load_smem_V_d =
      (tid % (kNumThreads / Bc)) *
      (kHeadDim / (kNumThreads / Bc)); // (tid % 2) * 32, 0,32,...
  // global Q row of current head for tile [Br,d] per block.
  int load_gmem_Q_Br = Q_tile_id * Br + load_smem_Q_Br; //当前block负责的全局Q行
  if (load_gmem_Q_Br >= QKV_seqlen)
    return;
  // KV tile gmem load index starts from 0 and increments with
  // each iteration as we loop over seqlen.
  int load_gmem_K_Bc_offset = 0;
  int load_gmem_V_Bc_offset = 0;

  // smem: Q | K×kStage（流水）| V。O 用寄存器+shuffle 写回，无单独 O smem。
  extern __shared__ half smem[];
  //每个Q tile共享内存的大小
  constexpr int Q_tile_size =
      Br * (kHeadDim + kPad); // 64*64=4096, ~8192 bytes=8M
  constexpr int KV_tile_size =
      Bc * (kHeadDim + kPad); // 64*64=4096, ~8192 bytes=8M
  // K multi-stages: currently, only apply multi stages for K across seq_len.
  half *Q_tile_smem = smem;                      // 8M/16M  Q_tile[Br][kHeadDim + kPad]
  half *K_tile_smem = Q_tile_smem + Q_tile_size; // 8M/16M  K_tile[kStage][Bc][kHeadDim + kPad]
  // K的smem按ping pong开
  half *V_tile_smem = K_tile_smem + kStage * KV_tile_size;//V_tile[Bc][kHeadDim + kPad]
  // head_dim多取64或128
  // TODO: KV may shared same smem to reduce smem usage for kStage 1
  // stage 2, no shared KV smem, Br=Bc=64,  d=64: 8M+(8M)*2+8M    =32M
  // stage 2, no shared KV smem, Br=Bc=64, d=128: 16M+(16M)*2+16M =64M
  // stage 2, no shared KV smem, Br=Bc=64, d=256: 32M+(32M)*2+32M =128M
  // stage 1, no shared KV smem, Br=Bc=64,  d=64: 8M+(8M)+8M      =24M
  // stage 1, no shared KV smem, Br=Bc=64, d=128: 16M+(16M)*1+16M =48M
  // stage 1, no shared KV smem, Br=Bc=32, d=256: 16M+(16M)*1+16M =48M
  
  //转成mma指令处理的指针
  uint32_t smem_Q_base_ptr = __cvta_generic_to_shared(Q_tile_smem);
  uint32_t smem_K_base_ptr = __cvta_generic_to_shared(K_tile_smem);
  uint32_t smem_V_base_ptr = __cvta_generic_to_shared(V_tile_smem);

  // 在线 softmax：每 lane 维护行的 m_old / l_old（float）；MMA 寄存器 R_S/R_O/R_D
  float lane_block_row_max_old[kWarpTileSeqLenQ][2]; // [1][2]
  float lane_block_row_sum_old[kWarpTileSeqLenQ][2]; // [1][2]
  fill_2D_regs<float, kWarpTileSeqLenQ, 2>(lane_block_row_max_old, -INFINITY);
  fill_2D_regs<float, kWarpTileSeqLenQ, 2>(lane_block_row_sum_old, 0.0f);

  // Registers for S=Q@K^T/O=P@V
  // registers for QKV, S=Q[Br,d]@K[Bc,d]=[Br,Bc]
  // and O=P[Br,Bc]@V[Bc,d]=[Br,d].
  //m16n8k16的mma，Q总共加载16x16到Tensor core的寄存器中，分为4个8x8的寄存器
  //k/v加载16x8，分为2个8x8的寄存器
  uint32_t R_Q[kWarpTileSeqLenQ][4];  // [1][4]  
  uint32_t R_K[kWarpTileSeqLenK][2];  // [8][2]
  uint32_t R_V[kWarpTileHeadDimV][2]; // [8][2]
  // registers for current tile_K_seqlen within, [64,64] = S_tile[Br,Bc]
  // = Q_tile[Br,d] * K[Bc,d], each thread hold 2x32 bits regs.
  uint32_t R_S[kWarpTileSeqLenQ][kWarpTileSeqLenK][2]; // [1][8][2]
  // registers for tile_K_seqlen O=PV[Br,d]=P@V, [2][2/4][2], 8 or 16 regs.
  // TODO: may reuse R_D as R_O? kWarpTileSeqLenP=kWarpTileSeqLenQ.
  uint32_t R_O[kWarpTileSeqLenP][kWarpTileHeadDimV][2]; // [1][8][2]
  // registers final Output [D]=final rescale(R_O), [2][2/4][2], 8 or 16 regs.
  uint32_t R_D[kWarpTileSeqLenP][kWarpTileHeadDimV][2]; // [1][8][2]
  fill_3D_regs<uint32_t, kWarpTileSeqLenQ, kWarpTileSeqLenK, 2>(R_S, 0);
  fill_3D_regs<uint32_t, kWarpTileSeqLenP, kWarpTileHeadDimV, 2>(R_D, 0);
  fill_3D_regs<uint32_t, kWarpTileSeqLenP, kWarpTileHeadDimV, 2>(R_O, 0);

  // Q：整块异步载入 smem（每个 block 一次）
  {
    int load_gmem_Q_d = load_smem_Q_d;
    //该偏移是针对全局内存Q，本block的当前thread负责的Q tile行在head_dim的第load_gmem_Q_d位置
    int load_gmem_Q_addr =
        (Q_gmem_offset + load_gmem_Q_Br * kHeadDim + load_gmem_Q_d);
    //该地址是共享内存smem_Q，偏移到Q tile中对应行在head_dim第load_gmem_Q_d位置
    uint32_t load_smem_Q_ptr =
        (smem_Q_base_ptr +
         (load_smem_Q_Br * (kHeadDim + kPad) + load_smem_Q_d) * sizeof(half));
#pragma unroll
    //每个线程沿d维连续负责kHeadDim / (kNumThreads / Br)个half，一次8个half一组调用async copy
    for (int i = 0; i < (kHeadDim / (kNumThreads / Br)); i += 8) {
      CP_ASYNC_CG(load_smem_Q_ptr + i * 2, &Q[load_gmem_Q_addr + i], 16); // 16Byte是8个half
    }
    //一次提交所有异步拷贝
    CP_ASYNC_COMMIT_GROUP();
  }

  // kStage>1：先填满除最后一槽外的 K buffer，便于与主循环流水衔接
  if constexpr (kStage > 1) {
#pragma unroll
    for (int stage = 0; stage < (kStage - 1); ++stage) {
      // update the offset of n according to stages
      load_gmem_K_Bc_offset = stage * Bc; // e.g (0~3)*64=(0,64,128,192,...)
      int load_gmem_K_Bc = load_gmem_K_Bc_offset + load_smem_K_Bc; // < seqlen
      int load_gmem_K_d = load_smem_K_d; // K [Bc,d] from [seqlen,d]
      int load_gmem_K_addr =
          (K_gmem_offset + load_gmem_K_Bc * kHeadDim + load_gmem_K_d);
      uint32_t load_smem_K_ptr =
          (smem_K_base_ptr +
           (stage * KV_tile_size + load_smem_K_Bc * (kHeadDim + kPad) +
            load_smem_K_d) *
               sizeof(half));
#pragma unroll
      for (int i = 0; i < (kHeadDim / (kNumThreads / Bc)); i += 8) {
        CP_ASYNC_CG(load_smem_K_ptr + i * 2, &K[load_gmem_K_addr + i], 16);
      }
      CP_ASYNC_COMMIT_GROUP();
    }
  }

  // 等待 Q 与已提交的 K 预取完成
  if constexpr (kStage > 1) {
    CP_ASYNC_WAIT_GROUP(kStage - 2); // s2->0, s3->1, s4->2
    __syncthreads();
  }

// 主循环：沿序列 K/V 按 Bc 分块；每步 S=QK^T -> softmax 得 P -> O+=PV，并在线重标定
#pragma unroll 1
  for (int tile_K_seqlen = 0; tile_K_seqlen < Tc; ++tile_K_seqlen) {
    // TODO: process last tile_K_seqlen ? pad to multiple of 8.
    // s2 tn 0->0, 1->1, 2->0; s3 tn 0->0, 1->1, 2->2, 3->0;
    int smem_sel = (tile_K_seqlen) % kStage;
    // s2 tn 0->1, 1->0, 2->1; s3 tn 0->2, 1->0, 2->1, 3->2;
    // 当tile_K_seqlen为0时，这里的ping pong buffer id是最后一个，对应上面的除最后一个外将所有buffer都填满
    int smem_sel_next = (tile_K_seqlen + (kStage - 1)) % kStage;

    // kStage>1：cp.async 预取当前 V、下一 K（双缓冲）；kStage==1 同步加载 K,V
    if constexpr (kStage > 1) {
      // First, prefetch curr V tile_K_seqlen [Bc,d] (no stages)
      {
        load_gmem_V_Bc_offset =
            tile_K_seqlen * Bc; // e.g (0~3)*64=(0,64,128,192,...)
        int load_gmem_V_Bc = load_gmem_V_Bc_offset + load_smem_V_Bc;
        int load_gmem_V_d = load_smem_V_d;
        int load_gmem_V_addr =
            (V_gmem_offset + load_gmem_V_Bc * kHeadDim + load_gmem_V_d);
        uint32_t load_smem_V_ptr =
            (smem_V_base_ptr +
             (load_smem_V_Bc * (kHeadDim + kPad) + load_smem_V_d) *
                 sizeof(half));
#pragma unroll
        for (int i = 0; i < (kHeadDim / (kNumThreads / Bc)); i += 8) {
          CP_ASYNC_CG(load_smem_V_ptr + i * 2, &V[load_gmem_V_addr + i], 16);
        }
        CP_ASYNC_COMMIT_GROUP();
      }

      // Then, prefetch next stage K (tile_K_seqlen + 1) [d,Bc]
      // stage为2时，对下一块K_tile做预取
      if ((tile_K_seqlen + 1) < Tc) {
        load_gmem_K_Bc_offset =
            (tile_K_seqlen + 1) * Bc; // e.g (0~3)*64=(0,64,128,192,...)
        int load_gmem_K_Bc = load_gmem_K_Bc_offset + load_smem_K_Bc; // < seqlen
        int load_gmem_K_d = load_smem_K_d; // K [Bc,d] from [seqlen,d]
        int load_gmem_K_addr =
            (K_gmem_offset + load_gmem_K_Bc * kHeadDim + load_gmem_K_d);
        uint32_t load_smem_K_ptr =
            (smem_K_base_ptr +
             (smem_sel_next * KV_tile_size +
              load_smem_K_Bc * (kHeadDim + kPad) + load_smem_K_d) *
                 sizeof(half));
#pragma unroll
        for (int i = 0; i < (kHeadDim / (kNumThreads / Bc)); i += 8) {
          CP_ASYNC_CG(load_smem_K_ptr + i * 2, &K[load_gmem_K_addr + i], 16);
        }
        CP_ASYNC_COMMIT_GROUP();
      }
      //没有双buffer，直接取本K和V的tile
    } else {
      // If no stages, kStage = 1, we have to load current K tile
      // from gmem to smem and have to wait it ready for Q@K^T MMA.

      // First, prefetch curr K tile_K_seqlen [d,Bc] (no stages)
      {
        load_gmem_K_Bc_offset =
            tile_K_seqlen * Bc; // e.g (0~3)*64=(0,64,128,192,...)
        int load_gmem_K_Bc = load_gmem_K_Bc_offset + load_smem_K_Bc; // < seqlen
        int load_gmem_K_d = load_smem_K_d; // K [Bc,d] from [seqlen,d]
        int load_gmem_K_addr =
            (K_gmem_offset + load_gmem_K_Bc * kHeadDim + load_gmem_K_d);
        uint32_t load_smem_K_ptr =
            (smem_K_base_ptr +
             (smem_sel * KV_tile_size + load_smem_K_Bc * (kHeadDim + kPad) +
              load_smem_K_d) *
                 sizeof(half));
#pragma unroll
        for (int i = 0; i < (kHeadDim / (kNumThreads / Bc)); i += 8) {
          CP_ASYNC_CG(load_smem_K_ptr + i * 2, &K[load_gmem_K_addr + i], 16);
        }

        CP_ASYNC_COMMIT_GROUP();
      }

      // Then, prefetch curr V tile_K_seqlen [d,Bc] (no stages)
      {
        load_gmem_V_Bc_offset =
            tile_K_seqlen * Bc; // e.g (0~3)*64=(0,64,128,192,...)
        int load_gmem_V_Bc = load_gmem_V_Bc_offset + load_smem_V_Bc;
        int load_gmem_V_d = load_smem_V_d;
        int load_gmem_V_addr =
            (V_gmem_offset + load_gmem_V_Bc * kHeadDim + load_gmem_V_d);
        uint32_t load_smem_V_ptr =
            (smem_V_base_ptr +
             (load_smem_V_Bc * (kHeadDim + kPad) + load_smem_V_d) *
                 sizeof(half));
#pragma unroll
        for (int i = 0; i < (kHeadDim / (kNumThreads / Bc)); i += 8) {
          CP_ASYNC_CG(load_smem_V_ptr + i * 2, &V[load_gmem_V_addr + i], 16);
        }
        CP_ASYNC_COMMIT_GROUP();
      }

      // Wait curr Q and K tile ready and let curr V tile copy async.
      CP_ASYNC_WAIT_GROUP(1);
      __syncthreads();
    }

    // S=Q@K^T：沿 d 以 kMmaAtomK=16 分块，LDM + HMMA16816 累加到 R_S
    fill_3D_regs<uint32_t, kWarpTileSeqLenQ, kWarpTileSeqLenK, 2>(R_S, 0);
#pragma unroll
    //这里的kMmaAtomK是一条mma能处理的k维元素数，不是Key的意思。kHeadDim / kMmaAtomK表示在k维要分多少个mma做计算
    //最后各个k维分块的mma结果要做acc
    for (int tile_K_d = 0; tile_K_d < (kHeadDim / kMmaAtomK); ++tile_K_d) {
#pragma unroll
      //kWarpTileSeqLenQ是一个warp负责该Q_tile块Br方向的几个mma块
      for (int i = 0; i < kWarpTileSeqLenQ; ++i) { // Q[Br,d]=[M,K]
        // 当前warp、i在Br方向上的偏移
        int warp_smem_Q_Br =
            warp_QP * (kMmaAtomM * kWarpTileSeqLenQ) + i * kMmaAtomM;
        // 每个lane负责的Br方向的偏移，2个线程负责同一行
        int lane_smem_Q_Br = warp_smem_Q_Br + lane_id % 16;            // 0~15
        // 在k dim方向的tiling，每个线程负责8个数
        int lane_smem_Q_d = tile_K_d * kMmaAtomK + (lane_id / 16) * 8; // 0,8
        uint32_t lane_smem_Q_ptr =
            (smem_Q_base_ptr +
             (lane_smem_Q_Br * (kHeadDim + kPad) + lane_smem_Q_d) *
                 sizeof(half));
        //上面的计算是为了满足LDMATRIX_X4要求的32个线程提供地址，每个线程提供8个half数的地址
        //将每个warp的Q_tile加载到tensor core的寄存器中
        LDMATRIX_X4(R_Q[i][0], R_Q[i][1], R_Q[i][2], R_Q[i][3],
                    lane_smem_Q_ptr); // now, R_Q
      }

#pragma unroll
      //kWarpTileSeqLenK通常为8，表示每个warp负责Bc方向的多少列
      //这里和Q不同，warp_KV为0，每个warp都需要加载全部Bc到自己的寄存器
      for (int j = 0; j < kWarpTileSeqLenK; ++j) {
        //计算出本循环在Bc方向的偏移，所有warp都相同
        int warp_smem_K_Bc =
            warp_KV * (kMmaAtomN * kWarpTileSeqLenK) + j * kMmaAtomN;
        // 在Bc当前的mma分块中，该线程对应的Bc行 0-7
        int lane_smem_K_Bc = warp_smem_K_Bc + lane_id % 8; // 0~7
        // 在k维对应的偏移量 thread 0~7,16~23 -> 0, thread 8~15,24~31 -> 8
        int lane_smem_K_d =
            tile_K_d * kMmaAtomK + ((lane_id / 8) % 2) * 8; // 0,8
        uint32_t lane_smem_K_ptr =
            (smem_K_base_ptr +
             (smem_sel * KV_tile_size + lane_smem_K_Bc * (kHeadDim + kPad) +
              lane_smem_K_d) *
                 sizeof(half));
        //LDMATRIX_X2只需要前16个thread提供地址，这里加载的时候没有使用trans
        //因为要计算Q@K^T，smem中存储的是K(行主序)，可以看做K^T的列主序
        LDMATRIX_X2(R_K[j][0], R_K[j][1], lane_smem_K_ptr); // R_K
      } // end for kWarpTileSeqLenK

#pragma unroll
      for (int i = 0; i < kWarpTileSeqLenQ; ++i) {
#pragma unroll
        for (int j = 0; j < kWarpTileSeqLenK; ++j) {
          HMMA16816(R_S[i][j][0], R_S[i][j][1], R_Q[i][0], R_Q[i][1], R_Q[i][2],
                    R_Q[i][3], R_K[j][0], R_K[j][1], R_S[i][j][0],
                    R_S[i][j][1]);
        }
      }
    } // end loop over d, S=Q@K^T，但是这里每个warp和每个kWarpTileSeqLenK的部分还没有reduce
    // S是[Br, Bc]的，Br分为warp部分，Bc分为kWarpTileSeqLenK部分
    // 要计算online softmax，只关心行的结果，所以warp间不需要reduce，只需要warp内对kWarpTileSeqLenK做reduce
    __syncthreads();

    // 本 tile 内 online softmax：行最大 -> exp 得 P 写入 R_S，warp 内归约 sum
    float lane_row_max_new[kWarpTileSeqLenQ][2]; // [1][2]
    float lane_row_sum_new[kWarpTileSeqLenQ][2]; // [1][2]
    fill_2D_regs<float, kWarpTileSeqLenQ, 2>(lane_row_max_new, -INFINITY);
    fill_2D_regs<float, kWarpTileSeqLenQ, 2>(lane_row_sum_new, 0.0f);

    // R_S 中 half2 布局对应 m16n8k16 的 C 片段（见 PTX 文档 Matrix Fragments）
    //求 tile块内每一行的最大值，即lane_row_max_new  
    //此处共4个warp128个thread，每个thread负责Br的2行，4个thread对应Br的一行数据
    //所以lane_row_max_new中有两行Br的数据，每4个thread就对应一个新的Br行
#pragma unroll
    for (int i = 0; i < kWarpTileSeqLenQ; ++i) {
#pragma unroll
      for (int j = 0; j < kWarpTileSeqLenK; ++j) {
        //详细内存排布参考mma笔记
        // 在一个warp内，共32个线程，每个线程的RS寄存器存储两个float16的数据，共64个数，组成一个8x8的矩阵
        // R_S[i][j][0]和R_S[i][j][1]是两个8x8的矩阵，上下排列，对应16x8的布局
        // 纵向16乘上4个warp，对应的是Br(64)，横向8(4个线程一行,每个线程2个数)乘上kWarpTileSeqLenK的8，对应的是Bc(64)
        // 对Bc做max reduce，只需关注warp内的数
        // 该循环内先对每个thread的kWarpTileSeqLenK * 2个数做max reduce
        float2 t_reg_S_0 = __half22float2(HALF2(R_S[i][j][0])); // 0~7  {c0, c1}
        float2 t_reg_S_1 = __half22float2(HALF2(R_S[i][j][1])); // 8~15 {c2, c3}
        // This should be the row max after S = (Q @ K^T) / sqrt(d)
        float tmp_max_0 = max(t_reg_S_0.x, t_reg_S_0.y) * scale;
        float tmp_max_1 = max(t_reg_S_1.x, t_reg_S_1.y) * scale;
        lane_row_max_new[i][0] = max(lane_row_max_new[i][0], tmp_max_0);
        lane_row_max_new[i][1] = max(lane_row_max_new[i][1], tmp_max_1);
      } // end for kWarpTileSeqLenK

      // Warp level reduce max, warp_size = 4
      // Each thread contains the maximum of 2 rows of Br,
      // and only the values of T0, T4, ..., T28 are used.
      // Br, row_id = warp_QP<0~3> * 32 + i<0> * 16 + 0 * 8 + (lane / 4) <0~7>

      //每个线程有两个Br行的数据，每4个线程是一行，上面已经求到每个线程内的局部max，这里在相邻的4个线程做max
      lane_row_max_new[i][0] =
          warp_reduce_max<float, 4>(lane_row_max_new[i][0]);
      // Br, row_id = warp_QP<0~3> * 32 + i<0> * 16 + 1 * 8 + (lane / 4) <8~15>
      lane_row_max_new[i][1] =
          warp_reduce_max<float, 4>(lane_row_max_new[i][1]);
    } // end for kWarpTileSeqLenQ
    __syncthreads();

    // 与历史 m 合并后算 exp(scale*S - m)，累加行和；R_S 存 P
#pragma unroll
    for (int i = 0; i < kWarpTileSeqLenQ; ++i) {
      //与历史max比较，更新历史max
      // Use latest global row max without update.
      // Br 0, row_id, 0~7,  16~23, 32~39, 48~55;
      float block_row_max_new_0 = lane_row_max_new[i][0];
      // Br 1, row_id, 8~15, 24~31, 40~47, 56~63;
      float block_row_max_new_1 = lane_row_max_new[i][1];

      float block_row_max_old_0 = lane_block_row_max_old[i][0];
      float block_row_max_old_1 = lane_block_row_max_old[i][1];
      // Apply m_new = max(m_old, m_new) here.
      block_row_max_new_0 = max(block_row_max_old_0, block_row_max_new_0);
      block_row_max_new_1 = max(block_row_max_old_1, block_row_max_new_1);

#pragma unroll
      for (int j = 0; j < kWarpTileSeqLenK; ++j) {
        float2 t_reg_S_0 = __half22float2(HALF2(R_S[i][j][0])); // 0~7  {c0, c1}
        float2 t_reg_S_1 = __half22float2(HALF2(R_S[i][j][1])); // 8~15 {c2, c3}
        // P = Exp(S - m_new), fmaf(x, y, z) = x * y + z;
        t_reg_S_0.x =
            __expf(__fmaf_rn(t_reg_S_0.x, scale, -block_row_max_new_0));
        t_reg_S_0.y =
            __expf(__fmaf_rn(t_reg_S_0.y, scale, -block_row_max_new_0));
        t_reg_S_1.x =
            __expf(__fmaf_rn(t_reg_S_1.x, scale, -block_row_max_new_1));
        t_reg_S_1.y =
            __expf(__fmaf_rn(t_reg_S_1.y, scale, -block_row_max_new_1));
        //计算新的行和
        lane_row_sum_new[i][0] += (t_reg_S_0.x + t_reg_S_0.y);
        lane_row_sum_new[i][1] += (t_reg_S_1.x + t_reg_S_1.y);
        // Update R_S for P[Br,Bc] = Exp(S-m), point wise.
        //将减过max的S写回R_S寄存器
        HALF2(R_S[i][j][0]) = __float22half2_rn(t_reg_S_0);
        HALF2(R_S[i][j][1]) = __float22half2_rn(t_reg_S_1);
      } // end for kWarpTileSeqLenK

      // Warp level reduce sum, warp_size = 4
      //在4个thread内对row sum求和
      lane_row_sum_new[i][0] =
          warp_reduce_sum<float, 4>(lane_row_sum_new[i][0]);
      lane_row_sum_new[i][1] =
          warp_reduce_sum<float, 4>(lane_row_sum_new[i][1]);
    } // end for kWarpTileSeqLenQ
    __syncthreads();

    // P@V：先等 V 进 smem，再 MMA；R_S 即 P 的 A 片段布局
    if constexpr (kStage > 1) {
      // NOTE: For kStage > 1, we have send V mem issues before K
      if ((tile_K_seqlen + 1) < Tc) {
        CP_ASYNC_WAIT_GROUP(1);
      } else {
        CP_ASYNC_WAIT_GROUP(0);
      }
    } else {
      CP_ASYNC_WAIT_GROUP(0);
    }
    __syncthreads();

    // 沿 Bc 维分块：LDM V（转置）+ HMMA；tile_V_Bc 决定取 R_S 中哪两段作 A
    fill_3D_regs<uint32_t, kWarpTileSeqLenP, kWarpTileHeadDimV, 2>(R_O, 0);
    //在计算P@V时，Bc是K维度，head_dim要分成多个mma的N
#pragma unroll
    for (int tile_V_Bc = 0; tile_V_Bc < (Bc / kMmaAtomK); ++tile_V_Bc) {
#pragma unroll
      for (int j = 0; j < kWarpTileHeadDimV; ++j) {
        int warp_smem_V_d = warp_KV * (kMmaAtomN * kWarpTileHeadDimV) +
                            j * kMmaAtomN; // d, matmaul N
        int lane_smem_V_Bc =
            tile_V_Bc * kMmaAtomK + lane_id % 16; // 0~15; Bc, matmul K
        int lane_smem_V_d = warp_smem_V_d;        // 0
        uint32_t lane_smem_V_ptr =
            (smem_V_base_ptr +
             (lane_smem_V_Bc * (kHeadDim + kPad) + lane_smem_V_d) *
                 sizeof(half));
        //注意，B矩阵要求列主序，而smem_v是行主序的，所以这里使用.trans的ldmatrix
        LDMATRIX_X2_T(R_V[j][0], R_V[j][1], lane_smem_V_ptr); // R_V
      }
      //因为R_S按Bc维度分成了多个kWarpTileSeqLenK部分，每个部分是一个16x8的。
      //A矩阵要是16x16的，两个R_S可以组成一个16x16的(横向堆叠)
      int w = tile_V_Bc * 2; // MMA(Warp) selected, 0, 2, 4, 6
#pragma unroll
      for (int i = 0; i < kWarpTileSeqLenP; ++i) { // 1
#pragma unroll
        for (int j = 0; j < kWarpTileHeadDimV; ++j) { // 8, 16, 32, ...
          HMMA16816(R_O[i][j][0], R_O[i][j][1], R_S[i][w][0], R_S[i][w][1],
                    R_S[i][w + 1][0], R_S[i][w + 1][1], R_V[j][0], R_V[j][1],
                    R_O[i][j][0], R_O[i][j][1]);
        }
      }
    } // end for V Bc.
    __syncthreads();

    // FA2 在线更新：D = exp(m_old-m)*D + O；l = exp(m_old-m)*l + sum(P)；再写回 m
    //由于每个线程负责两个Br行，所以计算流程中都是两份
#pragma unroll
    for (int i = 0; i < kWarpTileSeqLenP;
         ++i) { // kWarpTileSeqLenQ=kWarpTileSeqLenP=1
      // m = max(m_old, m_new), l = exp(m_old - m) * l_old + l_new (FA2 paper)
      // Br 0, row_id, 0~7,  16~23, 32~39, 48~55; Br 1, row_id, 8~15, 24~31,
      // 40~47, 56~63
      float block_row_max_new_0 = lane_row_max_new[i][0];
      float block_row_max_new_1 = lane_row_max_new[i][1];
      float block_row_sum_new_0 = lane_row_sum_new[i][0];
      float block_row_sum_new_1 = lane_row_sum_new[i][1];

      float block_row_max_old_0 = lane_block_row_max_old[i][0];
      float block_row_max_old_1 = lane_block_row_max_old[i][1];
      // NOTE: max(-inf, val) = val.
      block_row_max_new_0 = max(block_row_max_old_0, block_row_max_new_0);
      block_row_max_new_1 = max(block_row_max_old_1, block_row_max_new_1);
      // Avoid inf value while using m_old for rescaling O.
      block_row_max_old_0 =
          (tile_K_seqlen > 0 ? block_row_max_old_0 : block_row_max_new_0);
      block_row_max_old_1 =
          (tile_K_seqlen > 0 ? block_row_max_old_1 : block_row_max_new_1);

      // rescale factor for O and l, exp(m_old - m)
      //计算修正因子
      float rescale_o_factor_0 =
          __expf(block_row_max_old_0 - block_row_max_new_0);
      float rescale_o_factor_1 =
          __expf(block_row_max_old_1 - block_row_max_new_1);
#pragma unroll
      //对之前的结果做修正并求和，存到R_D中
      for (int j = 0; j < kWarpTileHeadDimV; ++j) { // 8, 16, 32, ...
        float2 t_reg_O_0 = __half22float2(HALF2(R_O[i][j][0])); // 0~7  {c0, c1}
        float2 t_reg_O_1 = __half22float2(HALF2(R_O[i][j][1])); // 8~15 {c2, c3}
        float2 t_reg_D_0 = __half22float2(HALF2(R_D[i][j][0])); // 0~7  {c0, c1}
        float2 t_reg_D_1 = __half22float2(HALF2(R_D[i][j][1])); // 8~15 {c2, c3}
        // Note that the formula in the FA2 paper is incorrect; here,
        // the inverse of the exp function should not be taken, as it
        // would result in an error during rescaling, namely, you have
        // use exp(m_old - m_new), not 1/(m_old - m_new).
        // O_new[Br,d] = exp(m_old - m_new) * O_old + P@V
        // fmaf(x, y, z) = x * y + z
        t_reg_D_0.x = __fmaf_rn(rescale_o_factor_0, t_reg_D_0.x, t_reg_O_0.x);
        t_reg_D_0.y = __fmaf_rn(rescale_o_factor_0, t_reg_D_0.y, t_reg_O_0.y);
        t_reg_D_1.x = __fmaf_rn(rescale_o_factor_1, t_reg_D_1.x, t_reg_O_1.x);
        t_reg_D_1.y = __fmaf_rn(rescale_o_factor_1, t_reg_D_1.y, t_reg_O_1.y);
        HALF2(R_D[i][j][0]) = __float22half2_rn(t_reg_D_0);
        HALF2(R_D[i][j][1]) = __float22half2_rn(t_reg_D_1);
      } // end for kWarpTileHeadDimV.
      
      //更新l和m
      // Now, we can update m, l after O has been scaled.
      // 1. First, update block row sum Exp for each lane which
      // need both m_new and m_old.
      float block_row_sum_old_0 = lane_block_row_sum_old[i][0];
      float block_row_sum_old_1 = lane_block_row_sum_old[i][1];
      // Update l = exp(m_old - m_new) * l_old + row_sum(P).
      lane_block_row_sum_old[i][0] = (__fmaf_rn(
          rescale_o_factor_0, block_row_sum_old_0, block_row_sum_new_0));
      lane_block_row_sum_old[i][1] = (__fmaf_rn(
          rescale_o_factor_1, block_row_sum_old_1, block_row_sum_new_1));
      // 2. Then, update block row max for each lane.
      lane_block_row_max_old[i][0] = block_row_max_new_0;
      lane_block_row_max_old[i][1] = block_row_max_new_1;
    }

    // NOTE: After compute P @ V, we have to wait next K tile ready in smem.
    // do not need to wait any things if kStage == 1.
    if constexpr (kStage > 1) {
      if ((tile_K_seqlen + 1) < Tc) {
        CP_ASYNC_WAIT_GROUP(0);
      }
      __syncthreads();
    }

  } // end loop over N
  __syncthreads();

  // 最终 O = D / l（分母为累积 softmax 归一化因子）
#pragma unroll
  for (int i = 0; i < kWarpTileSeqLenP; ++i) { // 1
    // __frcp_rn计算倒数
    float rescale_factor_0 = __frcp_rn(lane_block_row_sum_old[i][0]);
    float rescale_factor_1 = __frcp_rn(lane_block_row_sum_old[i][1]);
#pragma unroll
    //最终结果O都要除以l
    for (int j = 0; j < kWarpTileHeadDimV; ++j) {             // 8, 16, 32, ...
      float2 t_reg_D_0 = __half22float2(HALF2(R_D[i][j][0])); // 0~7  {c0, c1}
      float2 t_reg_D_1 = __half22float2(HALF2(R_D[i][j][1])); // 8~15 {c2, c3}
      t_reg_D_0.x = rescale_factor_0 * t_reg_D_0.x;
      t_reg_D_0.y = rescale_factor_0 * t_reg_D_0.y;
      t_reg_D_1.x = rescale_factor_1 * t_reg_D_1.x;
      t_reg_D_1.y = rescale_factor_1 * t_reg_D_1.y;
      HALF2(R_D[i][j][0]) = __float22half2_rn(t_reg_D_0);
      HALF2(R_D[i][j][1]) = __float22half2_rn(t_reg_D_1);
    }
  }

  // 128-bit 写回：lane shuffle 拼 fragment，st.global.v4
#pragma unroll
  for (int i = 0; i < kWarpTileSeqLenP; ++i) { // 1
#pragma unroll
    for (int j = 0; j < kWarpTileHeadDimV; ++j) { // 8，每4个thread负责同一Br行的8个fp16元素共128bit，8个循环则为head_dim总数据量
      uint32_t R_Z[2][4];                         // [2][4]
      //分别取4个thread的R_D存到R_Z中，拼成一个128bit的行
      R_Z[0][0] = R_D[i][j][0];
      R_Z[1][0] = R_D[i][j][1]; // warp_size 4
      R_Z[0][1] = __shfl_sync((0xffffffff), R_D[i][j][0], lane_id + 1, 4);
      R_Z[0][2] = __shfl_sync((0xffffffff), R_D[i][j][0], lane_id + 2, 4);
      R_Z[0][3] = __shfl_sync((0xffffffff), R_D[i][j][0], lane_id + 3, 4);
      R_Z[1][1] = __shfl_sync((0xffffffff), R_D[i][j][1], lane_id + 1, 4);
      R_Z[1][2] = __shfl_sync((0xffffffff), R_D[i][j][1], lane_id + 2, 4);
      R_Z[1][3] = __shfl_sync((0xffffffff), R_D[i][j][1], lane_id + 3, 4);

      // st.global.v4 128 bits. [Br,d]
      // 4个线程的 group的第一个lane负责写回global O
      if (lane_id % 4 == 0) {
        // (0/1)*32 + (0/1)*16=(0,16,32,48), + 0~7 -> 0~56
        //先计算出当前warp对应的在Q_tile内的Br行
        int store_warp_regs_O_Br =
            warp_QP * (kMmaAtomM * kWarpTileSeqLenP) + i * kMmaAtomM;
        // 计算出在global O中对应的Br行(每个4线程组的第一行)，O_tile_id是第几个Q_tile；一个warp对应16个Br行
        // lane_id取值只能为0 4 8 12 16 20 24 28
        int store_lane_gmem_O_Br =
            O_tile_id * Br + store_warp_regs_O_Br + lane_id / 4; // 0~7
        // (0~3)*16 + (0/1)*8=(0,8,16,24,...,48,56)
        //计算出在head_dim维度的偏移，每128bit的数只是head_dim的一部分，j是在head_dim方向的tiling
        int store_warp_regs_O_d =
            warp_KV * (kMmaAtomN * kWarpTileHeadDimV) + j * kMmaAtomN;
        int store_lane_gmem_O_d = store_warp_regs_O_d; // (0~3)*16+(0/8)
        int store_gmem_O_addr_0 =
            (O_gmem_offset + (store_lane_gmem_O_Br + 0) * kHeadDim +
             store_lane_gmem_O_d);
             //store_lane_gmem_O_Br+8即偏移到本4线程组负责的第二个Br行
        int store_gmem_O_addr_1 =
            (O_gmem_offset + (store_lane_gmem_O_Br + 8) * kHeadDim +
             store_lane_gmem_O_d);
        //使用float4写入
        LDST128BITS(O[store_gmem_O_addr_0]) = LDST128BITS(R_Z[0][0]);
        LDST128BITS(O[store_gmem_O_addr_1]) = LDST128BITS(R_Z[1][0]);
      }
    } // end for kWarpTileHeadDimV
  } // end for kWarpTileSeqLenQ
}

// Host：定 tile、算动态 smem、设属性、启动 kernel
template <const int kHeadDim, const int kStage>
void launch_flash_attn_mma_stages_split_q(torch::Tensor Q, torch::Tensor K,
                                          torch::Tensor V, torch::Tensor O) {
  // 与上方 kernel 模板实参一致；d<128 与 d>=128 仅 kWarpTileSeqLenK 等可能不同（见注释）。
  constexpr int kMmaAtomM = 16;
  constexpr int kMmaAtomN = 8;
  constexpr int kMmaAtomK = 16;
  constexpr int kMmaTileSeqLenQ = (kHeadDim < 128) ? 4 : 4; // 沿 Br 共 4 个 warp（split-Q）
  constexpr int kMmaTileSeqLenK = 1;                      // 与 kernel 断言一致
  constexpr int kMmaTileSeqLenP = (kHeadDim < 128) ? 4 : 4;
  constexpr int kMmaTileHeadDimV = 1;
  constexpr int kWarpTileSeqLenQ = 1;
  constexpr int kWarpTileSeqLenK = (kHeadDim < 128) ? 8 : 2; // Bc = 8*1*本值 → 64 或 16
  constexpr int kWarpTileSeqLenP = 1;
  constexpr int kWarpTileHeadDimV =
      (kHeadDim / (kMmaAtomN * kMmaTileHeadDimV)); // d/8，即沿 d 的子块个数
  constexpr int Br =
      kMmaAtomM * kMmaTileSeqLenQ * kWarpTileSeqLenQ; // 典型 16*4*1=64
  constexpr int Bc = kMmaAtomN * kMmaTileSeqLenK * kWarpTileSeqLenK;
  constexpr int kNumThreads =
      WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK;
  constexpr int kPad = 8;

  // static int kMaxSramPerBlock;
  // cudaDeviceGetAttribute(&kMaxSramPerBlock,
  // cudaDevAttrMaxSharedMemoryPerBlock, 0); Calculate SRAM size needed per
  // block, Q,K,V smem size
  const int smem_max_size =
      ((Br * (kHeadDim + kPad)) + (kStage * Bc * (kHeadDim + kPad)) +
       (Bc * (kHeadDim + kPad))) *
      sizeof(half);

  const int QKV_batch = Q.size(0);
  const int QKV_head = Q.size(1);
  const int QKV_seqlen = Q.size(2); // QKV_seqlen
  assert(QKV_seqlen % Bc == 0);     // multiple of Bc=64

  // TODO: How to apply block swizzle to improve L2 Cache hit rate?
  // NOTE: reorder (B,H,Tr) -> (Tr,B*H) seems can improve L2 Cache hit rate.
  // This might be because SM schedules blocks starting from the x-dimension.
  // Placing Tr at the forefront ensures that identical KV pairs are placed
  // in consecutive scheduling queues, thereby improving L2 Cache hit rates.
  // Tr(=N/Br), batch_size x num_heads
  dim3 grid(div_ceil(QKV_seqlen, Br), QKV_batch * QKV_head);
  dim3 block(kNumThreads); // 4/8 warps per block

  cudaFuncSetAttribute(
      flash_attn_mma_stages_split_q_kernel<
          kHeadDim, kMmaAtomM, kMmaAtomN, kMmaAtomK, kMmaTileSeqLenQ,
          kMmaTileSeqLenK, kMmaTileSeqLenP, kMmaTileHeadDimV, kWarpTileSeqLenQ,
          kWarpTileSeqLenK, kWarpTileSeqLenP, kWarpTileHeadDimV, kStage, kPad>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      // kMaxSramPerBlock
      98304);

  flash_attn_mma_stages_split_q_kernel<
      kHeadDim, kMmaAtomM, kMmaAtomN, kMmaAtomK, kMmaTileSeqLenQ,
      kMmaTileSeqLenK, kMmaTileSeqLenP, kMmaTileHeadDimV, kWarpTileSeqLenQ,
      kWarpTileSeqLenK, kWarpTileSeqLenP, kWarpTileHeadDimV, kStage, kPad>
      <<<grid, block, smem_max_size>>>(reinterpret_cast<half *>(Q.data_ptr()),
                                       reinterpret_cast<half *>(K.data_ptr()),
                                       reinterpret_cast<half *>(V.data_ptr()),
                                       reinterpret_cast<half *>(O.data_ptr()),
                                       QKV_seqlen, QKV_head);
}

// 入口：按 head_dim 与 stages 分发到模板实例
void flash_attn_mma_stages_split_q(torch::Tensor Q, torch::Tensor K,
                                   torch::Tensor V, torch::Tensor O,
                                   int stages) {
  CHECK_TORCH_TENSOR_DTYPE(Q, torch::kHalf) // Q [B,H,N,D]
  CHECK_TORCH_TENSOR_DTYPE(K, torch::kHalf) // K [B,H,N,D]
  CHECK_TORCH_TENSOR_DTYPE(V, torch::kHalf) // V [B,H,N,D]
  CHECK_TORCH_TENSOR_DTYPE(O, torch::kHalf) // O [B,H,N,D]
  const int d = Q.size(3);                  // B, H, N, d

  if (stages > 1) {
    switch (d) {
    case 32:
      launch_flash_attn_mma_stages_split_q<32, 2>(Q, K, V, O);
      break;
    case 64:
      launch_flash_attn_mma_stages_split_q<64, 2>(Q, K, V, O);
      break;
    case 96:
      launch_flash_attn_mma_stages_split_q<96, 2>(Q, K, V, O);
      break;
    case 128:
      launch_flash_attn_mma_stages_split_q<128, 2>(Q, K, V, O);
      break;
    default:
      throw std::runtime_error("headdim not support!");
      break;
    }
  } else {
    switch (d) {
    case 32:
      launch_flash_attn_mma_stages_split_q<32, 1>(Q, K, V, O);
      break;
    case 64:
      launch_flash_attn_mma_stages_split_q<64, 1>(Q, K, V, O);
      break;
    case 96:
      launch_flash_attn_mma_stages_split_q<96, 1>(Q, K, V, O);
      break;
    case 128:
      launch_flash_attn_mma_stages_split_q<128, 1>(Q, K, V, O);
      break;
    default:
      throw std::runtime_error("headdim not support!");
      break;
    }
  }
}
