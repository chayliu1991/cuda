#include <cstdio>

__global__ void helloWorldKernel()
{
	printf("hello world from gpu\n");
}

__global__ void helloWorldwithThreadKernel()
{
	printf("hello world from gpu block:%d thread£º%d \n",blockIdx.x,threadIdx.x);
}


int main()
{
 	printf("hello world from cpu\n");
    printf("--------------------------------------------------\n");

    helloWorldKernel<<<1,10>>>();
    cudaDeviceSynchronize();
    printf("--------------------------------------------------\n");

    helloWorldwithThreadKernel<<<4,2>>>();
    cudaDeviceSynchronize();
    printf("--------------------------------------------------\n");
    
    return 0;
}