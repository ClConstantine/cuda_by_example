#include "../common/book.h"
const int N = 33*1024;

__global__ void add(int *a, int *b, int *c){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    /**
        计算线程索引
     */

    while(tid < N){
        c[tid] = a[tid] + b[tid];
        tid += blockDim.x * gridDim.x;
        /*
            每次移动一个gird的大小
        */
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

    // add<<<(N + 127) / 128,128>>>(dev_a, dev_b, dev_c);
    /*
        有概率会出现错误，grid会超出限制
    */
    add<<<128,128>>>(dev_a, dev_b, dev_c);
    /*
        这里限制了grid的数量为128个，每个grid中有128个线程
        通过这种方式可以保证线程的数量不会超出限制
        在函数中我们通过while循环来保证所有数据都会被处理
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