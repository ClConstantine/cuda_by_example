/*
    CUDA流表示一个GPU操作队列,操作以指定顺序执行
*/

#include "../common/book.h"

const int N = 1024 * 1024;
const int FULL_DATA_SIZE = N * 200;

__global__ void kernel(int *a, int *b, int *c){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < N){
        int idx1 = (idx + 1) % 256;
        int idx2 = (idx + 2) % 256;

        float as = (a[idx] + a[idx1] + a[idx2]) / 3.0f;
        float bs = (b[idx] + b[idx1] + b[idx2]) / 3.0f;

        c[idx] = (as + bs) / 2;
    }
}

int main(){

    cudaDeviceProp prop;
    int whichDevice;
    HANDLE_ERROR(cudaGetDevice(&whichDevice));
    HANDLE_ERROR(cudaGetDeviceProperties(&prop,whichDevice));
    if(!prop.deviceOverlap){
        printf("Device will not handle overlaps, so no speed up from streams\n");
        return 0;
    }
    /*
        我们需要选择一个支持overlap的设备
        它在执行一个核函数的同时，可以复制数据到设备
        我们使用多个流来实现计算和数据复制的overlap

        这里是单个流的示例代码
    */

    cudaEvent_t start,stop;
    float elapsedTime;

    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    
    //初始化流
    cudaStream_t stream;
    HANDLE_ERROR(cudaStreamCreate(&stream));
    
    int *host_a,*host_b,*host_c;
    int *dev_a,*dev_b,*dev_c;

    HANDLE_ERROR(cudaMalloc((void**)&dev_a,N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b,N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_c,N * sizeof(int)));

    //注意这里分配的内存是页锁定内存
    HANDLE_ERROR(cudaHostAlloc((void**)&host_a,FULL_DATA_SIZE * sizeof(int),cudaHostAllocDefault));
    HANDLE_ERROR(cudaHostAlloc((void**)&host_b,FULL_DATA_SIZE * sizeof(int),cudaHostAllocDefault));
    HANDLE_ERROR(cudaHostAlloc((void**)&host_c,FULL_DATA_SIZE * sizeof(int),cudaHostAllocDefault));
    

    for(int i = 0;i < FULL_DATA_SIZE;i++){
        host_a[i] = rand();
        host_b[i] = rand();
    }

    /*
        这里我们不选择将全部数据复制到设备，
        而是选择每次复制N个数据进行分块计算.
        
        cudaMemcpyAsync()将以异步的方式复制数据到设备
        任何传递给cudaMemcpyAsync()的主机指针都必须是页锁定内存
        如果你将常规的主机内存传递给 cudaMemcpyAsync()，
        会使复制过程退化为同步操作，从而失去异步执行的优势。

        异步函数的行为是在流中执行放置一个请求
        当函数返回时，请求可能还没有被执行
        存在的保证是复制操作在下一个流中的操作前执行

        注意这里的kernel函数是在流中执行的
        此时的函数调用是异步的

        我们能够保证流的操作是按照传入的顺序执行的
    */
    HANDLE_ERROR(cudaEventRecord(start,0));

    for(int i = 0;i < FULL_DATA_SIZE;i += N){
        HANDLE_ERROR(cudaMemcpyAsync(dev_a,host_a + i,N * sizeof(int),cudaMemcpyHostToDevice,stream));
        HANDLE_ERROR(cudaMemcpyAsync(dev_b,host_b + i,N * sizeof(int),cudaMemcpyHostToDevice,stream));

        kernel<<<N/256,256,0,stream>>>(dev_a,dev_b,dev_c);

        HANDLE_ERROR(cudaMemcpyAsync(host_c + i,dev_c,N * sizeof(int),cudaMemcpyDeviceToHost,stream));
    }

    /*
        当for循环结束时，我们需要等待所有的流操作完成
        调用cudaStreamSynchronize()来等待流中的所有操作完成
    */
    HANDLE_ERROR(cudaStreamSynchronize(stream));

    HANDLE_ERROR(cudaEventRecord(stop,0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime,start,stop));

    printf("Time taken: %3.1f ms\n",elapsedTime);
    //Time taken: 247.2 ms

    HANDLE_ERROR(cudaFreeHost(host_a));
    HANDLE_ERROR(cudaFreeHost(host_b));
    HANDLE_ERROR(cudaFreeHost(host_c));
    
    HANDLE_ERROR(cudaFree(dev_a));
    HANDLE_ERROR(cudaFree(dev_b));
    HANDLE_ERROR(cudaFree(dev_c));

    HANDLE_ERROR(cudaStreamDestroy(stream));

    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));

    return 0;
}