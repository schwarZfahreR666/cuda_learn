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
    //获取线程块内数据起始地址 
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
	int *idata = g_idata + blockIdx.x*blockDim.x;
	//in-place reduction in global memory
    //循环规约一个block中的元素，相邻配对
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
//从线程分化角度或者合并访存角度看，都是最好的
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


int main(int argc,char** argv)
{
	initDevice(0);
	
	bool bResult = false;
	//initialization

	int size = 1 << 24;
	printf("	with array size %d  ", size);

	//execution configuration
	int blocksize = 1024;
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