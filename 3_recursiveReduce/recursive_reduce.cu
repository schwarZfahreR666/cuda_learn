#include <cuda_runtime.h>
#include "../include/utils.h"

int recursiveReduce_cpu(int *data, int const size)
{
	// terminate check
	if (size == 1) return data[0];
	// renew the stride
    //二分数据
	int const stride = size / 2;
	if (size % 2 == 1)
	{
		for (int i = 0; i < stride; i++)
		{
			data[i] += data[i + stride];
		}
        //奇数项无人配对时单独处理
		data[0] += data[size - 1];
	}
	else
	{
		for (int i = 0; i < stride; i++)
		{
			data[i] += data[i + stride];
		}
	}
	// call
	return recursiveReduce_cpu(data, stride);
}

__global__ void warmup(int * g_idata, int * g_odata, unsigned int n)
{
	//set thread ID
	unsigned int tid = threadIdx.x;
	//boundary check
	if (tid >= n) return;
	//convert global data pointer to the
    //为了索引不乱，获取线程块内数据起始地址 
	int *idata = g_idata + blockIdx.x*blockDim.x;
	//in-place reduction in global memory
    //循环规约一个block中的元素，相邻配对
	for (int stride = 1; stride < blockDim.x; stride *= 2)
	{
		if ((tid % (2 * stride)) == 0)
		{
            //相隔stride的两个元素规约
			idata[tid] += idata[tid + stride];
		}
		//synchronize within block
		__syncthreads();
	}
	//write result for this block to global mem
    //写到块索引对应的输出内存中
	if (tid == 0)
		g_odata[blockIdx.x] = idata[0];

}
//相邻配对规约
__global__ void reduceNeighbored(int * g_idata,int * g_odata,unsigned int n) 
{
	//set thread ID
	unsigned int tid = threadIdx.x;
	//boundary check
	if (tid >= n) return;
	//convert global data pointer to the
    //获取线程块内数据起始地址 
	//最好给每个block做一个内存起始指针，以防止索引混乱
	int *idata = g_idata + blockIdx.x*blockDim.x;
	//in-place reduction in global memory
    //循环规约一个block中的元素，相邻配对
	/*
	以thread_0为例分析行为
		thread_0 += thread_1
		thread_0 += thread_2
		thread_0 += thread_4
		 ...
	但是要考虑线程块中所有的线程如何分化
	*/
	for (int stride = 1; stride < blockDim.x; stride *= 2)
	{
		if ((tid % (2 * stride)) == 0) //每次规约相邻的线程都不会同时工作，会有分化
		{
            //相隔stride的两个元素规约
			idata[tid] += idata[tid + stride];
		}
		//synchronize within block
		__syncthreads();
	}
	//write result for this block to global mem
    //写到块索引对应的输出内存中
	if (tid == 0)
		g_odata[blockIdx.x] = idata[0];

}
//为防止分化，把要计算的数据映射到相邻的线程
__global__ void reduceNeighboredLess(int * g_idata,int *g_odata,unsigned int n)
{
	unsigned int tid = threadIdx.x;
    //拿到全局线程id
	unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
	// convert global data pointer to the local point of this block
    //内存块对应内存的起始地址
	int *idata = g_idata + blockIdx.x*blockDim.x;
	if (idx > n)
		return;
	//in-place reduction in global memory
	for (int stride = 1; stride < blockDim.x; stride *= 2)
	{
		//convert tid into local array index
        //把需要规约的数据id映射到连续的线程id上
        //让tid向后移动到需要规约的数据处
		int index = 2 * stride *tid;
        /* (tid/index[A,B])    
        第一轮：0/[0, 1] 1/[2, 3] 2/[4, 5] 3/[6, 7]
        第二轮：0/[0, 2] 1/[4, 6]
        第三轮：0/[0, 4]
        */
		if (index < blockDim.x)
		{
			idata[index] += idata[index + stride];
		}
		__syncthreads();
	}
	//write result for this block to global men
	if (tid == 0)
		g_odata[blockIdx.x] = idata[0];
}
//交错规约
//从线程分化角度或者合并访存(一个warp中所有线程同时访问的内存地址是连续的，
//可以在一次访存中连续读出多个数)角度看，都是最好的
//但是实际实验结果与理论不符
__global__ void reduceInterleaved(int * g_idata, int *g_odata, unsigned int n)
{
	unsigned int tid = threadIdx.x;
	unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
	// convert global data pointer to the local point of this block
	int *idata = g_idata + blockIdx.x*blockDim.x;
	if (idx >= n)
		return;
	//in-place reduction in global memory
    //交错配对规约
	for (int stride = blockDim.x/2; stride >0; stride >>=1)
	{
		
		if (tid <stride)
		{
			idata[tid] += idata[tid + stride];
		}
		__syncthreads();
	}
	//write result for this block to global men
	if (tid == 0)
		g_odata[blockIdx.x] = idata[0];
}

/*使用共享内存*/
//相邻配对规约，共享内存
__global__ void reduceNeighbored_shared(int * g_idata,int * g_odata,unsigned int n) 
{
	//使用动态共享内存
	extern __shared__ int shared_m[];
	//set thread ID
	unsigned int tid = threadIdx.x;
	//boundary check
	if (tid >= n) return;
	//convert global data pointer to the
    //获取线程块内数据起始地址 
	//最好给每个block做一个内存起始指针，以防止索引混乱
	int *idata = g_idata + blockIdx.x*blockDim.x;
	//每个线程搬运一个数
	shared_m[tid] = idata[tid];
	__syncthreads();
	//in-place reduction in global memory
    //循环规约一个block中的元素，相邻配对
	/*
	以thread_0为例分析行为
		thread_0 += thread_1
		thread_0 += thread_2
		thread_0 += thread_4
		 ...
	但是要考虑线程块中所有的线程如何分化
	*/
	for (int stride = 1; stride < blockDim.x; stride *= 2)
	{
		if ((tid % (2 * stride)) == 0) //每次规约相邻的线程都不会同时工作，会有分化
		{
            //相隔stride的两个元素规约
			shared_m[tid] += shared_m[tid + stride];
		}
		//synchronize within block
		__syncthreads();
	}
	//write result for this block to global mem
    //写到块索引对应的输出内存中
	if (tid == 0)
		g_odata[blockIdx.x] = shared_m[0];

}

//跳跃取数减少bank冲突
__global__ void reduceNeighbored_shared_skip(int * g_idata,int *g_odata,unsigned int n)
{
	//使用动态共享内存
	extern __shared__ int shared_m[];
	unsigned int tid = threadIdx.x;
    //拿到全局线程id
	unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
	// convert global data pointer to the local point of this block
    //内存块对应内存的起始地址
	int *idata = g_idata + blockIdx.x*blockDim.x;
	if (idx > n)
		return;
	//每个线程搬运一个数
	shared_m[tid] = idata[tid];
	__syncthreads();
	//in-place reduction in global memory
	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{
		//convert tid into local array index
        //把需要规约的数据id映射到连续的线程id上
        //让tid向后移动到需要规约的数据处
		int index = tid;
        /* (tid/index[A,B]) 
		只看同一个warp中是否会有bank冲突,bank冲突要看同一条指令(不同指令在不同时钟周期，是不同的内存事务)   
        第一轮：0/[0, 1] 1/[2, 3] 2/[4, 5] 3/[6, 7] ... 16/[32, 33](32与0 bank冲突)
        第二轮：0/[0, 2] 1/[4, 6] ... 8/[32, 34](与0冲突)
        第三轮：0/[0, 4]

		所以改成每个线程去访问tid和tid + reduce_num / 2的元素
		在读取第一个数的时候，每个线程访问的都是自己tid位置的共享内存
		同一线程束的访存都在共享内存同一行
        */
		if (index < stride)
		{
			shared_m[index] += shared_m[index + stride];
		}
		__syncthreads();
	}
	//write result for this block to global men
	if (tid == 0)
		g_odata[blockIdx.x] = shared_m[0];
}

//减少idle线程
__global__ void reduceNeighbored_shared_skip_halfthread(int * g_idata,int *g_odata,unsigned int n)
{
	//使用动态共享内存
	extern __shared__ int shared_m[];
	unsigned int tid = threadIdx.x;
    //拿到全局线程id，偶数block把奇数block的数据也包了，少启动一半线程
	unsigned idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	// convert global data pointer to the local point of this block
    //内存块对应内存的起始地址
	int *idata = g_idata + blockIdx.x * (blockDim.x * 2);

	int *idata_next = idata + blockDim.x;
	if (idx > n)
		return;
	//每个线程搬运一个数，搬数的时候就把相应位置的加法计算出来
	shared_m[tid] = idata[tid] + idata_next[tid];
	__syncthreads();
	//in-place reduction in global memory
	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{
		//convert tid into local array index
        //把需要规约的数据id映射到连续的线程id上
        //让tid向后移动到需要规约的数据处
		int index = tid;
        /* (tid/index[A,B]) 
		只看同一个warp中是否会有bank冲突,bank冲突要看同一条指令(不同指令在不同时钟周期，是不同的内存事务)   
        第一轮：0/[0, 1] 1/[2, 3] 2/[4, 5] 3/[6, 7] ... 16/[32, 33](32与0 bank冲突)
        第二轮：0/[0, 2] 1/[4, 6] ... 8/[32, 34](与0冲突)
        第三轮：0/[0, 4]

		所以改成每个线程去访问tid和tid + reduce_num / 2的元素
		在读取第一个数的时候，每个线程访问的都是自己tid位置的共享内存
		同一线程束的访存都在共享内存同一行
        */
		if (index < stride)
		{
			shared_m[index] += shared_m[index + stride];
		}
		__syncthreads();
	}
	//write result for this block to global men
	if (tid == 0)
		g_odata[blockIdx.x] = shared_m[0];
}

//当展开reduce的循环
__device__ void warpReduce(volatile int* cache,int tid){
    cache[tid]+=cache[tid+32];
    cache[tid]+=cache[tid+16];
    cache[tid]+=cache[tid+8];
    cache[tid]+=cache[tid+4];
    cache[tid]+=cache[tid+2];
    cache[tid]+=cache[tid+1];
}

//循环展开
/*
一个warp中的32个线程其实是在一个SIMD单元上，这32个线程每次都是执行同一条指令，这天然地保持了同步状态，
因而当s=32时，即只有一个SIMD单元在工作时，完全可以将__syncthreads()这条同步代码去掉。
所以我们将最后一维进行展开以减少同步。
*/
__global__ void reduceNeighbored_shared_skip_halfthread_unroll(int * g_idata,int *g_odata,unsigned int n)
{
	//使用动态共享内存
	extern __shared__ int shared_m[];
	unsigned int tid = threadIdx.x;
    //拿到全局线程id，偶数block把奇数block的数据也包了，少启动一半线程
	unsigned idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	// convert global data pointer to the local point of this block
    //内存块对应内存的起始地址
	int *idata = g_idata + blockIdx.x * (blockDim.x * 2);

	int *idata_next = idata + blockDim.x;
	if (idx > n)
		return;
	//每个线程搬运一个数
	shared_m[tid] = idata[tid] + idata_next[tid];
	__syncthreads();
	//in-place reduction in global memory
	for (int stride = blockDim.x / 2; stride > 32; stride >>= 1)
	{
		//convert tid into local array index
        //把需要规约的数据id映射到连续的线程id上
        //让tid向后移动到需要规约的数据处
		int index = tid;
        /* (tid/index[A,B]) 
		只看同一个warp中是否会有bank冲突,bank冲突要看同一条指令(不同指令在不同时钟周期，是不同的内存事务)   
        第一轮：0/[0, 1] 1/[2, 3] 2/[4, 5] 3/[6, 7] ... 16/[32, 33](32与0 bank冲突)
        第二轮：0/[0, 2] 1/[4, 6] ... 8/[32, 34](与0冲突)
        第三轮：0/[0, 4]

		所以改成每个线程去访问tid和tid + reduce_num / 2的元素
		在读取第一个数的时候，每个线程访问的都是自己tid位置的共享内存
		同一线程束的访存都在共享内存同一行
        */
		if (index < stride)
		{
			shared_m[index] += shared_m[index + stride];
		}
		__syncthreads();
	}
	//循环展开
	if (tid < 32) warpReduce(shared_m, tid);
	//write result for this block to global men
	if (tid == 0)
		g_odata[blockIdx.x] = shared_m[0];
}

//使用shuffle指令
/*
洗牌指令
第一个参数，是一个32位的bit mask，对应一个线程内的32个线程束，mask为1代表该线程参与运算
最后一个参数w，代表逻辑上的线程束大小为w
其它参数视指令有所不同
*/
__device__ __forceinline__ int warpReduceSum(int sum, unsigned int blockSize){
	/*
	__shfl_down_sync(mask, v, d, w)
	标号为t的参与线程获得标号为t+d的线程中变量v的值，
	本质上把通道 ID 较高的线程中的变量广播到了 ID 较低的线程
	当调用 __shfl_down_sync(0xffffffff, x, 2) 时，默认 width 等于 warpSize（即32），则标号为 
	0-29的线程分别获得标号为2-31的线程中变量 x 的值；当调用 __shfl_down_sync(mask, x, 2, 16)时，
	warp 被划分为两个宽度为16的子线程组，则标号为0-13的线程分别获得标号为2-15的线程中变量 x 的值，
	warp 内通道 ID 为16-29的线程（在 warp 的子线程组内标号为0-13）
	分别获得通道ID为18-31的线程中变量 x 的值。
	*/
	//在每个线程束内做计算
	// t[0] += t[16] ... t[15] += t[31]
    if(blockSize >= 32)sum += __shfl_down_sync(0xffffffff,sum,16);
	// t[0] += t[8] ... t[7] += t[15]
    if(blockSize >= 16)sum += __shfl_down_sync(0xffffffff,sum,8);
	// t[0] += t[4] ... t[3] += t[7]
    if(blockSize >= 8)sum += __shfl_down_sync(0xffffffff,sum,4);
	// t[0] += t[2] t[1] += t[3]
    if(blockSize >= 4)sum += __shfl_down_sync(0xffffffff,sum,2);
	// t[0] += t[1]
    if(blockSize >= 2)sum += __shfl_down_sync(0xffffffff,sum,1);
	//等于做了线程束内的reduce
    return sum;
}
__global__ void reduce_with_shuffle(int * g_idata,int *g_odata,int n){
    int sum = 0;
	#define WARP_SIZE (32)
    //each thread loads one element from global memory to shared mem
    unsigned int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    unsigned int tid = threadIdx.x;
	//计算相邻block每个thread对应数据的和到sum
    sum += g_idata[idx] + g_idata[idx + blockDim.x];
	

    // shared mem for partial sums(one per warp in the block
    static __shared__ float warpLevelSums[WARP_SIZE];
    const int laneId = threadIdx.x % WARP_SIZE;
    const int warpId = threadIdx.x / WARP_SIZE;
	//由于每个线程块最多1024个线程，故这个数不会大于32
	const int warp_per_block = blockDim.x / WARP_SIZE; 

    sum = warpReduceSum(sum, /*blockSize*/blockDim.x);
	//这步计算后，每WARP_SIZE个线程对应的和存在sum中，结果存在每个线程束对应的共享内存内
    if(laneId == 0)warpLevelSums[warpId]=sum;
    __syncthreads();

	//把每个warp的sum分给tid小于warp_per_block的线程，继续做reduce
	//比如warp_per_block为32，那么前32个线程继续做reduce
	//tid为1的thread，sum是之前第1个线程束reduce得到的sum
    sum = (threadIdx.x < warp_per_block)? warpLevelSums[laneId]:0;
    // 最后做warp_per_block次reduce
    if(warpId == 0) sum = warpReduceSum(sum, /*blockSize*/warp_per_block);

    if(tid==0)g_odata[blockIdx.x]=sum;
}

int main(int argc,char** argv)
{
	initDevice(0);
	
	bool bResult = false;
	//initialization

	int size = 1 << 24;
	printf("	with array size %d  ", size);

	//execution configuration
	unsigned int blocksize = 1024;
	if (argc > 1)
	{
		blocksize = atoi(argv[1]);
	}
	dim3 block(blocksize, 1);
	dim3 grid((size - 1) / block.x + 1, 1);
	printf("grid %d block %d \n", grid.x, block.x);

	//allocate host memory
	size_t bytes = size * sizeof(int);
	int *idata_host = (int*)malloc(bytes);
	int *odata_host = (int*)malloc(grid.x * sizeof(int));
	int * tmp = (int*)malloc(bytes);

	//initialize the array
	initialData_int(idata_host, size);

	memcpy(tmp, idata_host, bytes);
	double iStart, iElaps;
	int gpu_sum = 0;

	// device memory
	int * idata_dev = NULL;
	int * odata_dev = NULL;
	CHECK(cudaMalloc((void**)&idata_dev, bytes));
	CHECK(cudaMalloc((void**)&odata_dev, grid.x * sizeof(int)));

	//cpu reduction
	int cpu_sum = 0;
	iStart = cpuSecond();
	//cpu_sum = recursiveReduce_cpu(tmp, size);
    //直接累加
	for (int i = 0; i < size; i++)
		cpu_sum += tmp[i];
	printf("cpu sum:%d \n", cpu_sum);
	iElaps = cpuSecond() - iStart;
	printf("cpu reduce                 elapsed %lf ms cpu_sum: %d\n", iElaps, cpu_sum);


	//kernel 1:reduceNeighbored
	
	CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	iStart = cpuSecond();
	warmup <<<grid, block >>>(idata_dev, odata_dev, size);
	cudaDeviceSynchronize();
	iElaps = cpuSecond() - iStart;
	cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
	gpu_sum = 0;
    //每个块的结果CPU加在一起
	for (int i = 0; i < grid.x; i++)
		gpu_sum += odata_host[i];
	printf("gpu warmup                 elapsed %lf ms gpu_sum: %d<<<grid %d block %d>>>\n",
		iElaps, gpu_sum, grid.x, block.x);

	//kernel 1:reduceNeighbored
	/*
	Section: GPU Speed Of Light Throughput
    ----------------------- ------------- ------------
    Metric Name               Metric Unit Metric Value
    ----------------------- ------------- ------------
    DRAM Frequency          cycle/nsecond         8.24
    SM Frequency            cycle/nsecond         1.83
    Elapsed Cycles                  cycle    4,439,672
    Memory Throughput                   %        25.19
    DRAM Throughput                     %        19.26
    Duration                      msecond         2.43
    L1/TEX Cache Throughput             %        22.17
    L2 Cache Throughput                 %        25.19
    SM Active Cycles                cycle    4,278,664
    Compute (SM) Throughput             %        39.11
    ----------------------- ------------- ------------
	*/
	CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	iStart = cpuSecond();
	reduceNeighbored << <grid, block >> >(idata_dev, odata_dev, size);
	cudaDeviceSynchronize();
	iElaps = cpuSecond() - iStart;
	cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
	gpu_sum = 0;
	for (int i = 0; i < grid.x; i++)
		gpu_sum += odata_host[i];
	printf("gpu reduceNeighbored       elapsed %lf ms gpu_sum: %d<<<grid %d block %d>>>\n",
		iElaps, gpu_sum, grid.x, block.x);

	//kernel 2:reduceNeighboredLess
	/*
	Section: GPU Speed Of Light Throughput
    ----------------------- ------------- ------------
    Metric Name               Metric Unit Metric Value
    ----------------------- ------------- ------------
    DRAM Frequency          cycle/nsecond         8.24
    SM Frequency            cycle/nsecond         1.83
    Elapsed Cycles                  cycle    2,474,394
    Memory Throughput                   %        44.48
    DRAM Throughput                     %        34.56
    Duration                      msecond         1.35
    L1/TEX Cache Throughput             %        39.69
    L2 Cache Throughput                 %        44.48
    SM Active Cycles                cycle 2,317,393.25
    Compute (SM) Throughput             %        27.41
    ----------------------- ------------- ------------
	*/
	CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	iStart = cpuSecond();
	reduceNeighboredLess <<<grid, block>>>(idata_dev, odata_dev, size);
	cudaDeviceSynchronize();
	iElaps = cpuSecond() - iStart;
	cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
	gpu_sum = 0;
	for (int i = 0; i < grid.x; i++)
		gpu_sum += odata_host[i];
	printf("gpu reduceNeighboredLess   elapsed %lf ms gpu_sum: %d<<<grid %d block %d>>>\n",
		iElaps, gpu_sum, grid.x, block.x);

	//kernel 3:reduceInterleaved
	/*
	Section: GPU Speed Of Light Throughput
    ----------------------- ------------- ------------
    Metric Name               Metric Unit Metric Value
    ----------------------- ------------- ------------
    DRAM Frequency          cycle/nsecond         8.24
    SM Frequency            cycle/nsecond         1.83
    Elapsed Cycles                  cycle    2,157,387
    Memory Throughput                   %        28.97
    DRAM Throughput                     %        28.97
    Duration                      msecond         1.18
    L1/TEX Cache Throughput             %         9.70
    L2 Cache Throughput                 %        10.87
    SM Active Cycles                cycle    1,998,483
    Compute (SM) Throughput             %        26.34
    ----------------------- ------------- ------------
	*/
	CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	iStart = cpuSecond();
	reduceInterleaved << <grid, block >> >(idata_dev, odata_dev, size);
	cudaDeviceSynchronize();
	iElaps = cpuSecond() - iStart;
	cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
	gpu_sum = 0;
	for (int i = 0; i < grid.x; i++)
		gpu_sum += odata_host[i];
	printf("gpu reduceInterleaved      elapsed %lf ms gpu_sum: %d<<<grid %d block %d>>>\n",
		iElaps, gpu_sum, grid.x, block.x);

	CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	iStart = cpuSecond();
	reduceNeighbored_shared << <grid, block , blocksize * sizeof(int)>> >(idata_dev, odata_dev, size);
	cudaDeviceSynchronize();
	iElaps = cpuSecond() - iStart;
	cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
	gpu_sum = 0;
	for (int i = 0; i < grid.x; i++)
		gpu_sum += odata_host[i];
	printf("gpu reduceNeighbored_shared      elapsed %lf ms gpu_sum: %d<<<grid %d block %d>>>\n",
		iElaps, gpu_sum, grid.x, block.x);

	CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	iStart = cpuSecond();
	reduceNeighbored_shared_skip << <grid, block , blocksize * sizeof(int)>> >(idata_dev, odata_dev, size);
	cudaDeviceSynchronize();
	iElaps = cpuSecond() - iStart;
	cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
	gpu_sum = 0;
	for (int i = 0; i < grid.x; i++)
		gpu_sum += odata_host[i];
	printf("gpu reduceNeighbored_shared_skip  elapsed %lf ms gpu_sum: %d<<<grid %d block %d>>>\n",
		iElaps, gpu_sum, grid.x, block.x);
	
	dim3 block_half(blocksize, 1);
	dim3 grid_half((size - 1) / (block_half.x * 2) + 1, 1);
	CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	iStart = cpuSecond();
	reduceNeighbored_shared_skip_halfthread << <grid_half, block_half , blocksize * sizeof(int)>> >(idata_dev, odata_dev, size);
	cudaDeviceSynchronize();
	iElaps = cpuSecond() - iStart;
	cudaMemcpy(odata_host, odata_dev, grid_half.x * sizeof(int), cudaMemcpyDeviceToHost);
	gpu_sum = 0;
	for (int i = 0; i < grid_half.x; i++)
		gpu_sum += odata_host[i];
	printf("gpu reduceNeighbored_shared_skip_halfthread  elapsed %lf ms gpu_sum: %d<<<grid %d block %d>>>\n",
		iElaps, gpu_sum, grid_half.x, block_half.x);


	CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	iStart = cpuSecond();
	reduceNeighbored_shared_skip_halfthread_unroll << <grid_half, block_half , blocksize * sizeof(int)>> >(idata_dev, odata_dev, size);
	cudaDeviceSynchronize();
	iElaps = cpuSecond() - iStart;
	cudaMemcpy(odata_host, odata_dev, grid_half.x * sizeof(int), cudaMemcpyDeviceToHost);
	gpu_sum = 0;
	for (int i = 0; i < grid_half.x; i++)
		gpu_sum += odata_host[i];
	printf("gpu reduceNeighbored_shared_skip_halfthread_unroll  elapsed %lf ms gpu_sum: %d<<<grid %d block %d>>>\n",
		iElaps, gpu_sum, grid_half.x, block_half.x);

	CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	iStart = cpuSecond();
	reduce_with_shuffle<<<grid_half, block_half , blocksize * sizeof(int)>>>(idata_dev, odata_dev, size);
	cudaDeviceSynchronize();
	iElaps = cpuSecond() - iStart;
	cudaMemcpy(odata_host, odata_dev, grid_half.x * sizeof(int), cudaMemcpyDeviceToHost);
	gpu_sum = 0;
	for (int i = 0; i < grid_half.x; i++)
		gpu_sum += odata_host[i];
	printf("gpu reduce_with_shuffle  elapsed %lf ms gpu_sum: %d<<<grid %d block %d>>>\n",
		iElaps, gpu_sum, grid_half.x, block_half.x);
	// free host memory

	free(idata_host);
	free(odata_host);
	CHECK(cudaFree(idata_dev));
	CHECK(cudaFree(odata_dev));

	//reset device
	cudaDeviceReset();

	//check the results
	if (gpu_sum == cpu_sum)
	{
		printf("Test success!\n");
	}
	return EXIT_SUCCESS;

}