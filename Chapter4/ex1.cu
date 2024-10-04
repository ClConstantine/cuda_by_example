#include "../common/book.h"

const int N = 10;

__global__ void add(int *a, int *b, int *c){
    int tid = blockIdx.x;
    /**
        blockIdx是一个内置变量
        包含当前Device代码执行的线程块的索引
        blockIdx是一个dim3类型的变量，包含了一个x,y,z三个成员,支持多维的线程块索引
        但是在这里我们只使用了x成员
     */ 

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

    add<<<N,1>>>(dev_a, dev_b, dev_c);
    /*
        第一个参数gridDim,用来指定函数执行时的线程块的数量。
        第二个参数blockDim,用来指定每个线程块中的线程数量。
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