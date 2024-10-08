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

int main(){

    unsigned char *buffer = (unsigned char*)big_random_block(SIZE);

    clock_t start, stop;
    start = clock();

    int *histo = (int*)std::malloc(sizeof(int) * 256);

    for(int i = 0;i < SIZE; i++){
        histo[buffer[i]]++;
    }
    stop = clock();
    float elapsedTime = (float)(stop - start) / (float)CLOCKS_PER_SEC * 1000.0f;
    printf( "Time to generate:  %3.1f ms\n", elapsedTime );
    //81.8ms

    long histoCount = 0;
    for(int i=0; i<256; i++){
        histoCount += histo[i];
    }
    std::cout << "Histogram Sum: " << histoCount << std::endl;

    free(buffer);

    return 0;
}