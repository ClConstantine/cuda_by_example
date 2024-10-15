/*
	零拷贝内存
	在cudaHostAlloc中，我们可以通过cudaHostAllocMapped来分配内存
	device可以直接访问这块内存，而不需要通过cudaMemcpy
	故称为零拷贝内存
	
    但是零拷贝内存会占用系统的物理内存，最终会降低系统性能
    当输入内存和输出内存都只使用一次时，使用零拷贝内存会提高性能
    但是如果输入内存和输出内存都会被多次使用，对于PCIE总线上的优化不如拷贝，最终使得性能下降
*/

#include "../common/book.h"
#include <algorithm> 
#include <cstdlib>
#include <iostream>

const int N = 1024 * 1024 * 50;
const int ROUND = 10;
const int threadsPerBlock = 256;
const int blocksPerGrid = std::min(32,(N + threadsPerBlock - 1) / threadsPerBlock);

__global__ void dot(int size, float *a, float *b, float *c){
    __shared__ float cache[threadsPerBlock];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    float temp = 0;
    while(tid < size){
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    cache[cacheIndex] = temp;

    __syncthreads();

    int i = blockDim.x / 2;
    while(i != 0){
        if(cacheIndex < i){
            cache[cacheIndex] += cache[cacheIndex + i];
            
        }
        __syncthreads();
        i /= 2;
    }
    if(cacheIndex == 0){
        c[blockIdx.x] = cache[0];
    }
}

float malloc_test(int size){
    cudaEvent_t start,stop;
    float elapsedTime;
    float *a,*b,*partial_c;
    float *dev_a,*dev_b,*dev_partial_c;

    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));

    a = (float*)malloc(size * sizeof(float));
    b = (float*)malloc(size * sizeof(float));
    partial_c = (float*)malloc(blocksPerGrid * sizeof(float));

    HANDLE_ERROR(cudaMalloc((void**)&dev_a,size * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b,size * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_partial_c,blocksPerGrid * sizeof(float)));

    for(int i = 0; i < size; i++){
        a[i] = i;
        b[i] = i * 2;
    }

    HANDLE_ERROR(cudaEventRecord(start,0));

    HANDLE_ERROR(cudaMemcpy(dev_a,a,size * sizeof(float),cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b,b,size * sizeof(float),cudaMemcpyHostToDevice));

    dot<<<blocksPerGrid,threadsPerBlock>>>(size,dev_a,dev_b,dev_partial_c);
    //cudaMemcpy()会隐式同步
    HANDLE_ERROR(cudaMemcpy(partial_c,dev_partial_c,blocksPerGrid * sizeof(float),cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaEventRecord(stop,0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime,start,stop));

    float c = 0;
    for(int i = 0; i < blocksPerGrid; i++){
        c += partial_c[i];
    }
    HANDLE_ERROR(cudaFree(dev_a));
    HANDLE_ERROR(cudaFree(dev_b));
    HANDLE_ERROR(cudaFree(dev_partial_c));
    
    free(a);
    free(b);
    free(partial_c);

    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));

    // printf("Value calculated: %f\n",c);

    return elapsedTime;
}

float cuda_host_alloc_test(int size){
    cudaEvent_t start,stop;
    float elapsedTime;
    float *a,*b,*partial_c;
    float *dev_a,*dev_b,*dev_partial_c;

    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));

    //cudaHostAllocMapped标志的内存可以让gpu直接访问    
    HANDLE_ERROR(cudaHostAlloc((void**)&a,size * sizeof(float),cudaHostAllocMapped | cudaHostAllocWriteCombined));
    HANDLE_ERROR(cudaHostAlloc((void**)&b,size * sizeof(float),cudaHostAllocMapped | cudaHostAllocWriteCombined));
    HANDLE_ERROR(cudaHostAlloc((void**)&partial_c,blocksPerGrid * sizeof(float),cudaHostAllocMapped | cudaHostAllocWriteCombined));

    for(int i = 0;i < size;i++){
        a[i] = i;
        b[i] = i * 2;
    }
    /*
        由于gpu的虚拟内存空间和cpu的虚拟内存空间是不同的
        所以需要通过cudaHostGetDevicePointer来获取gpu的指针
    */
    HANDLE_ERROR(cudaHostGetDevicePointer(&dev_a,a,0));
    HANDLE_ERROR(cudaHostGetDevicePointer(&dev_b,b,0));
    HANDLE_ERROR(cudaHostGetDevicePointer(&dev_partial_c,partial_c,0));

    HANDLE_ERROR(cudaEventRecord(start,0));
    dot<<<blocksPerGrid,threadsPerBlock>>>(size,dev_a,dev_b,dev_partial_c);
    HANDLE_ERROR(cudaThreadSynchronize());

    HANDLE_ERROR(cudaEventRecord(stop,0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime,start,stop));

    float c = 0;
    for(int i = 0; i < blocksPerGrid; i++){
        c += partial_c[i];
    }

    HANDLE_ERROR(cudaFreeHost(a));
    HANDLE_ERROR(cudaFreeHost(b));
    HANDLE_ERROR(cudaFreeHost(partial_c));

    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));

    // printf("Value calculated: %f\n",c);
    return elapsedTime;
}

int main(){
   
    cudaDeviceProp prop;
    int whichDevice;
    HANDLE_ERROR(cudaGetDevice(&whichDevice));
    HANDLE_ERROR(cudaGetDeviceProperties(&prop,whichDevice));
    if(prop.canMapHostMemory != 1){
        printf("Device cannot map memory.\n");
        return 0;
    }

    float elapsedTime;

    elapsedTime = 0;
    for(int i = 0; i < ROUND; i++){
        elapsedTime += malloc_test(N);
    }
    printf("Time using malloc: %3.1f ms\n",elapsedTime / ROUND);
    // Time using malloc: 45.7 ms
    
    elapsedTime = 0;
    for(int i = 0; i < ROUND; i++){
        elapsedTime += cuda_host_alloc_test(N);
    }
    printf("Time using cudaHostAlloc: %3.1f ms\n",elapsedTime / ROUND);
    // Time using cudaHostAlloc: 34.9 ms

    return 0;
}
