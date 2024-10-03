#include <stdio.h>
#include "../common/book.h"

__global__ void add(int a,int b,int* c) {
    *c = a + b;
}
__global__ void mul(){
    
}

int main(){

    int c;
    int *dev_c;
    HANDLE_ERROR(cudaMalloc((void**)&dev_c,sizeof(int)));
    /*
        cudaMalloc(void** devPtr,int size)在device上分配内存。
        devPtr用于指向保存在device新分配内存地址的变量
        返回void*,注意 CUDA 虽然淡化了host代码和device代码的差异
        但我们不能在host代码中对返回的指针进行Dereference
        绝对不能在host直接读取or写入
    */

    add<<<1,1>>>(2,7,dev_c);
    /*
        可以将dev_c传给device/host上执行的函数
        可以在device code里对dev_c指向空间进行读写
        不可以在主机上对dev_c指向空间进行读写
    */    
    HANDLE_ERROR(cudaMemcpy(&c,dev_c,sizeof(int),cudaMemcpyDeviceToHost));
    /*
        可以通过cudaMemcpy拷贝device的内存
        还有
            cudaMemcpyDeviceToHost() 
            cudaMemcpyDeviceToDevice()
        等等
    */

    printf("2 + 7 = %d\n",c);
    cudaFree(dev_c);
    
    /*
        查询CUDA设备
    */
    int cuda_count;
    HANDLE_ERROR(cudaGetDeviceCount(&cuda_count));

    printf("CUDA device: %d\n", cuda_count);

    for(int i = 0;i < cuda_count;i++){
        cudaDeviceProp prop;
        HANDLE_ERROR(cudaGetDeviceProperties(&prop,i));
        printf("Device %s has compute capability %d.%d.\n",prop.name,prop.major,prop.minor);
        printf("Overlap: %s\n",prop.deviceOverlap?"Yes":"No");
        printf("multiprocessor count: %d\n",prop.multiProcessorCount);
    }

    
    return 0;
}