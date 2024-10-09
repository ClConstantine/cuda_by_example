/*
    malloc()和cudaHostAlloc()的区别
    malloc()分配的内存是可分页的，
    cudaHostAlloc()分配的页锁定内存

    页锁定内存不回将内存页交换到磁盘上，
    操作系统可以安全的访问物理地址

    由于知道物理地址，那么可以使用DMA复制数据
    从主机到设备，而不需要CPU参与，所以将内存固定很重要

    事实上CUDA对于可分页数据的处理仍然是通过DMA的方式
    所以会将数据复制到一块临时锁定内存，再复制到GPU

    但是不能将每一个malloc()替换为cudaHostAlloc()
    因为cudaHostAlloc()分配的内存是固定的，会很快消耗完memory

    建议仅对cudaMemcpy()的源和目的地址使用cudaHostAlloc()
    在不使用时立刻free.
*/

#include "../common/book.h"

const int SIZE = 1024 * 1024 * 10;

float cuda_malloc_test(int size,bool up){
    cudaEvent_t start,stop;
    int *a,*dev_a;
    float elapsedTime;

    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));

    a = (int*)malloc(size*sizeof(int));
    HANDLE_NULL(a);
    HANDLE_ERROR(cudaMalloc((void**)&dev_a,size*sizeof(int)));

    HANDLE_ERROR(cudaEventRecord(start,0));

    for(int i=0;i<100;i++){
        if(up){
            HANDLE_ERROR(cudaMemcpy(dev_a,a,size*sizeof(int),cudaMemcpyHostToDevice));
        }else{
            HANDLE_ERROR(cudaMemcpy(a,dev_a,size*sizeof(int),cudaMemcpyDeviceToHost));
        }
    }

    HANDLE_ERROR(cudaEventRecord(stop,0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime,start,stop));

    free(a);
    HANDLE_ERROR(cudaFree(dev_a));
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));

    return elapsedTime;
}

float cuda_host_alloc_test(int size,bool up){
    cudaEvent_t start,stop;
    int *a,*dev_a;
    float elapsedTime;

    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));

    HANDLE_ERROR(cudaHostAlloc((void**)&a,size*sizeof(int),cudaHostAllocDefault));
    /*
        分配主机锁定内存
    */
    HANDLE_ERROR(cudaMalloc((void**)&dev_a,size*sizeof(int)));

    HANDLE_ERROR(cudaEventRecord(start,0));

    for(int i=0;i<100;i++){
        if(up){
            HANDLE_ERROR(cudaMemcpy(dev_a,a,size*sizeof(int),cudaMemcpyHostToDevice));
        }else{
            HANDLE_ERROR(cudaMemcpy(a,dev_a,size*sizeof(int),cudaMemcpyDeviceToHost));
        }
    }

    HANDLE_ERROR(cudaEventRecord(stop,0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime,start,stop));

    HANDLE_ERROR(cudaFreeHost(a));
    HANDLE_ERROR(cudaFree(dev_a));
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));

    return elapsedTime;
}



int main(){

    float elapsedTime;
    elapsedTime = cuda_malloc_test(SIZE,true);
    printf("Time using cudaMalloc: %3.1f ms\n",elapsedTime);

    elapsedTime = cuda_malloc_test(SIZE,false);
    printf("Time using cudaMalloc: %3.1f ms\n",elapsedTime);

    elapsedTime = cuda_host_alloc_test(SIZE,true);
    printf("Time using cudaHostAlloc: %3.1f ms\n",elapsedTime);

    elapsedTime = cuda_host_alloc_test(SIZE,false);
    printf("Time using cudaHostAlloc: %3.1f ms\n",elapsedTime);

    /*
        10MB
        Time using cudaMalloc: 398.3 ms
        Time using cudaMalloc: 447.9 ms
        Time using cudaHostAlloc: 354.6 ms
        Time using cudaHostAlloc: 390.4 ms

        100MB
        Time using cudaMalloc: 4062.6 ms
        Time using cudaMalloc: 4659.0 ms
        Time using cudaHostAlloc: 3442.6 ms
        Time using cudaHostAlloc: 3298.2 ms
    */




    return 0;
}
