/*
    原子操作
    计算集版本1.2以上才能完全支持,当然4060都8.9了

    计算集合编译开关 -arch=sm_12
    4060可以 -arch=sm_89

    对于 x++, 包含了3个步骤
    1. 读取x的值
    2. 将x的值加1
    3. 将新值写入x
    这个过程我们称为读-改-写

    如果我们在多个线程中同时执行这个操作，那么就会出现竞争条件
    我们在多线程中需要保证在完成这3ge步骤的过程中，不会被其他线程打断
    这就是原子操作的作用
*/
#include <cstdlib>
#include <iostream>
#include "../common/book.h"

#define SIZE (100*1024*1024)

__global__ void histo_kernel(unsigned char* buffer, unsigned int* histo, long size){
    int offest = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    // 偏移量 & 步长

    while(offest < size){
        atomicAdd(&(histo[buffer[offest]]), 1);
        /*atomicAdd(addr, y)*/
        offest += stride;
    }

}

int main(){

    unsigned char *buffer = (unsigned char*)big_random_block(SIZE);

    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start, 0));
    
    unsigned char *dev_buffer;
    unsigned int *dev_histo;
    HANDLE_ERROR(cudaMalloc((void**)&dev_buffer, SIZE));
    HANDLE_ERROR(cudaMemcpy(dev_buffer, buffer, SIZE, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMalloc((void**)&dev_histo, 256*sizeof(int)));
    HANDLE_ERROR(cudaMemset(dev_histo, 0, 256*sizeof(int)));

    cudaDeviceProp prop;
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));


    int blocks = prop.multiProcessorCount;
    histo_kernel<<<blocks * 2, 256>>>(dev_buffer, dev_histo, SIZE);
    unsigned int histo[256];
    HANDLE_ERROR(cudaMemcpy(histo, dev_histo, 256 * sizeof(int), cudaMemcpyDeviceToHost));


    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    float elapsedTime;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("Time to generate: %3.1f ms\n", elapsedTime);
    // 610.4ms
    /*
        由于可能产生大量竞争,为了保证原子性,
        相同位置的造作被串行化了,队列非常长,性能极差
    */

    long histoCount = 0;
    for(int i=0; i<256; i++){
        histoCount += histo[i];
    }
    std::cout << "Histogram Sum: " << histoCount << std::endl;

    for(int i = 0;i < SIZE; i++){
        histo[buffer[i]]--;
    }
    for(int i = 0;i < 256; i++){
        if(histo[i] != 0){
            std::cout << "Failure at " << i << "!" << std::endl;
        }
    }

    free(buffer);
    HANDLE_ERROR(cudaFree(dev_buffer));
    HANDLE_ERROR(cudaFree(dev_histo));
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));

    return 0;
}