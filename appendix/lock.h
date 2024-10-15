
struct Lock {
    int* mutex;

    Lock() {
        HANDLE_ERROR(cudaMalloc((void**)&mutex, sizeof(int)));
        HANDLE_ERROR(cudaMemset(mutex, 0, sizeof(int)));
    }

    ~Lock() {
        cudaFree(mutex);
    }

    __device__ void lock() {
        while (atomicCAS(mutex, 0, 1) != 0);
        __threadfence();
        /*
            GPU 通常会采用弱内存排序，它允许处理器以不同于程序代码的顺序执行内存操作
            这意味着线程可能不会按顺序看到其他线程的内存操作。
            为了确保线程看到的内存操作是按顺序执行的，
            需要在内存操作之间插入 __threadfence() 调用。
        */
    
    }
    __device__ void unlock() {
        atomicExch(mutex, 0);
        __threadfence();
    }
};
