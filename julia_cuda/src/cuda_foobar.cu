#include <stdio.h>

__global__ void i_print_my_tid()
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello from thread %d!\n", tid);
    printf("  blockIdx.x = %d, blockDim.x = %d, threadIdx.x = %d\n", blockIdx.x, blockDim.x, threadIdx.x);
}

int main ()
{
    // 2 blocks of 4 threads each 
    i_print_my_tid<<<3, 7>>>();
    cudaDeviceSynchronize(); 
    return 0;
}