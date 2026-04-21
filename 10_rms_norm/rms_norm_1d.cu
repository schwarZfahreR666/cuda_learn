#include <cuda_runtime.h>

template<const int kWARP_SIZE=32>
__device__ __forceinline__ float warp_reduce_sum(float val){
    #pragma unroll
    for(int mask = kWARP_SIZE >> 1; mask >= 1; mask >>= 1){
        val += __shfl_xor_sync(0xffffffff, val, mask, kWARP_SIZE);
    }

    return val;
}

template<const int kThreadNums=256, const int kWARP_SIZE=32>
__device__ __forceinline__ float block_reduce_sum(float val){
    const int kWarpNums = (kThreadNums + kWARP_SIZE - 1) / kWARP_SIZE;
    int warp_id = threadIdx.x / kWARP_SIZE;
    int lane_id = threadIdx.x % kWARP_SIZE;

    val = warp_reduce_sum(val);
    __shared__ float smem[kWarpNums];
    if(lane_id == 0) smem[warp_id] = val;
    __syncthreads();

    if(lane_id < kWarpNums) val = smem[lane_id];
    else val = 0.0f;

    val = warp_reduce_sum(val);

    return val;
}

template<const int kThreads = 256>
__global__ void rms_norm(const float* input, float gamma, float beta, float* output, int N,
                      float eps){
    //N threads.  1 block

    int NTiles = (N + kThreads - 1) / kThreads;
    int tx = threadIdx.x;
    float val = 0.0f;
    float sum = 0.0f;
    for(int n = 0; n < NTiles; ++n){
        int idx = n * kThreads + tx;
        val = idx < N ? input[idx] : 0.0f;
        float variance = val * val;
        sum += block_reduce_sum<kThreads>(variance);
    }

    float r_rms = rsqrt(sum / N + eps);

    for(int n = 0; n < NTiles; ++n){
        int idx = n * kThreads + tx;
        val = idx < N ? input[idx] : 0.0f;

        float ans = val * r_rms * gamma + beta;

        if(idx < N) output[idx] = ans;
    }

}

// input, output are device pointers
extern "C" void solve(const float* input, float gamma, float beta, float* output, int N,
                      float eps) {
    constexpr int kThreads = 256;
    dim3 block(kThreads);
    dim3 grid(1);

    rms_norm<kThreads><<<grid, block>>>(input, gamma, beta, output, N, eps);


}