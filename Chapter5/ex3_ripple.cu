#include "../common/image.h"
#include "../common/book.h"
#include <iostream>

using std::cin;using std::cout;using std::endl;     //NOLINT

const int DIM = 1024;

__global__ void kernel(unsigned char* ptr, int ticks){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int offset = x + y * gridDim.x * blockDim.x;

    float fx = x - DIM/2;
    float fy = y - DIM/2;
    float d = sqrtf(fx * fx + fy * fy);
    unsigned char grey = (unsigned char)(128.0f + 127.0f * cos(d/10.0f - ticks/7.0f) / (d/10.0f + 1.0f));

    ptr[offset*4 + 0] = grey;
    ptr[offset*4 + 1] = grey;
    ptr[offset*4 + 2] = grey;
    ptr[offset*4 + 3] = 255;
}

int main(){
    IMAGE bitmap(DIM,DIM);
    unsigned char *device_bitmap;
    HANDLE_ERROR(cudaMalloc((void**)&device_bitmap,bitmap.image_size()));

    dim3 blocks(DIM / 16,DIM / 16);
    dim3 threads(16,16);

    int ticks = 0;

    while(1){
        kernel<<<blocks,threads>>>(device_bitmap,ticks);
        HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(),device_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost));
        
        ticks++;
        char key = bitmap.show_image(30);
        if(key==27) break;

    }

    cudaFree(device_bitmap);

    return 0;
}
