#include <iostream>
#include "kernel.hpp"

extern int global_block_x, global_block_y;

__global__ void julia_kernel_gpu(float *julia_set, Complex c, float scale, int res_x, int res_y, int max_iter, float max_mag, float x_scale, float y_scale){
    //we no longer pass in the pixel coords but calculate them from the thread parameters
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = res_x * res_y;
    //guard against too many thread launches 
    if (tid >= total_pixels) return;

    int x = tid / res_y; 
    int y = tid % res_y;
    julia_set[x * res_y + y] = (float)(tid % 256);
}   



void julia_kernel(float *julia_set, Complex c, float scale, int res_x, int res_y, int max_iter, float max_mag, float x_scale, float y_scale) {

    int total_pixels = res_x * res_y;
    // compute a good default block size
    int block_size = 256; //multiple of 32 (warp size) 
    if (global_block_x > 0) {
        block_size = global_block_x;
    }

    //array size 
    size_t mem_size = total_pixels * sizeof(float);
    float *d_julia_set;
    cudaMalloc(&d_julia_set, mem_size);

    int grid_size = (total_pixels + block_size - 1) / block_size; //round up division

    julia_kernel_gpu<<<grid_size,block_size>>>(d_julia_set, c, scale, res_x, res_y, max_iter, max_mag, x_scale, y_scale);
    cudaDeviceSynchronize();
    cudaMemcpy(julia_set, d_julia_set, mem_size, cudaMemcpyDeviceToHost);
    cudaFree(d_julia_set);
    

}
