#include<stdio.h>
__global__ void hello_world(void)
{
  printf("GPU: Hello world!\n");
}
int main(int argc,char **argv)
{
  printf("CPU: Hello world!\n");
  hello_world<<<1,10>>>();
  // cudaDeviceReset();//包含了隐式同步
  cudaDeviceSynchronize();//显示同步也可以
  return 0;
}