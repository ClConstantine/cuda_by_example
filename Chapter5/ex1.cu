#include "../common/book.h"

const int N = 10;

__global__ void add(int *a, int *b, int *c){
    int tid = threadIdx.x;
    /**
        threadIdx是一个内置变量
        包含当前Device代码执行的线程的索引
     */ 
    
    // int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // printf("blockIdx.x = %d, blockDim.x = %d, threadIdx.x = %d\n", blockIdx.x, blockDim.x, threadIdx.x);
    // 高维向量的线程索引

    if(tid < N){
        c[tid] = a[tid] + b[tid];
    }
}

int main(){

    int a[N], b[N], c[N];
    int *dev_a, *dev_b, *dev_c;

    HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_c, N * sizeof(int)));


    for(int i = 0;i < N;i++){
        a[i] = -i;
        b[i] = i * i;
    }

    HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice));

    add<<<1,N>>>(dev_a, dev_b, dev_c);
    /*
        这里我们修改为了add<<<1,N>>>...
        在1个线程块中有N个线程进行并行线程的执行
        线程限制为maxThreadsPerBlock的值
    
        blockDim可以是一个三维量
    */

    HANDLE_ERROR(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));

    for(int i = 0;i < N;i++){
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);


    return 0;
}