#include "../common/book.h"
#include "lock.h"
#include <algorithm> 
#include <iostream>

const int N = 33 * 1024 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = std::min(32,(N + threadsPerBlock - 1) / threadsPerBlock);
/*
    最多32个block，每个block中有256个线程
    当N较小时，不需要太多block
*/

__global__ void dot(Lock lock, double *a, double *b, double *c){
    __shared__ double cache[threadsPerBlock];
    /*
        共享内存（shared memory）是每个block共享的
        __shared__关键字告诉编译器这个变量是共享内存中的变量
        通过共享内存来减少全局内存的访问
     */

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    double temp = 0;
    while(tid < N){
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    cache[cacheIndex] = temp;

    __syncthreads();
    /*
        线程同步，等待所有线程都执行完毕
    */

    int i = blockDim.x / 2;
    /*
        这里要求是2的幂次方
        线程复用，把每个block算出来的
    */
    while(i != 0){
        if(cacheIndex < i){
            cache[cacheIndex] += cache[cacheIndex + i];
            
        }
        __syncthreads();
        /* 
            __syncthreads()要求线程块中的所有线程都必须到达这一点并等待，
            否则会出现线程发散的问题，CUDA架构要求确保每个线程都能到达同步点。
            如果有一些线程到达了,而其他线程没有到达，这会导致程序崩溃或不确定的行为。
        */
        i /= 2;
    }
    if(cacheIndex == 0){
        lock.lock();
        c[0] += cache[0];
        lock.unlock();
    }
}

int main(){
    double *a, *b, c = 0;
    double *dev_a, *dev_b, *dev_c;

    a = (double*)malloc(N * sizeof(double));
    b = (double*)malloc(N * sizeof(double));

    HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_c, sizeof(double)));

    for(int i = 0;i < N;i++){
        a[i] = i;
        b[i] = i * 2;
    }


    HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemset(dev_c, 0, sizeof(double)));

    dot<<<blocksPerGrid,threadsPerBlock>>>(Lock{},dev_a, dev_b, dev_c);

    HANDLE_ERROR(cudaMemcpy(&c, dev_c, sizeof(double), cudaMemcpyDeviceToHost));
    
    std::cout<<c<<std::endl;

    #define sum_squares(x)  (x*(x+1)*(2*x+1)/6)
    printf( "Does GPU value %.6g = %.6g?\n", c,
             2 * sum_squares( (double)(N - 1) ) );

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    free(a);
    free(b);
    
    return 0;
}
