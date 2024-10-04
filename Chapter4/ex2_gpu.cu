#include "../common/book.h"
#include "../common/image.h"

const int DIM = 1000;

struct cuComplex{
    float r;
    float i;
    __device__ cuComplex(float a,float b):r(a),i(b){}

    __device__ float magnitude2(void){
        return r*r + i*i;
    }
    __device__ cuComplex operator*(const cuComplex &a){
        return cuComplex(r*a.r - i*a.i,i*a.r + r*a.i);
    }
    __device__ cuComplex operator+(const cuComplex &a){
        return cuComplex(r+a.r,i+a.i);
    }
};

__device__ int julia(int x,int y){
    const float scale = 1.5;
    float jx = scale * (float)(DIM/2 - x)/(DIM/2);
    float jy = scale * (float)(DIM/2 - y)/(DIM/2);

    cuComplex c(-0.8,0.156);
    cuComplex a(jx,jy);

    int i = 0;
    for(i=0;i<200;i++){
        a = a*a + c;
        if(a.magnitude2() > 1000)
            return 0;
    }
    return 1;
}

__global__ void kernel(unsigned char *ptr){
    int x = blockIdx.x;
    int y = blockIdx.y;

    int offset = x + y*gridDim.x;
    int juliaValue = julia(x,y);
    /*
        从device调用的函数需要加上__device__修饰符
    */

    ptr[offset*4 + 0] = 255 * juliaValue;
    ptr[offset*4 + 1] = 0;
    ptr[offset*4 + 2] = 0;
    ptr[offset*4 + 3] = 255;

}

int main(){
    IMAGE bitmap(DIM,DIM);
    unsigned char *device_bitmap;

    HANDLE_ERROR(cudaMalloc((void**)&device_bitmap,bitmap.image_size()));

    dim3 grid(DIM,DIM);
    kernel<<<grid,1>>>(device_bitmap);
    /*
        打包了一个三维的dim值
        传给kernel函数，这样就可以使用x,y,z的方式来访问线程了
    */
    HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(),device_bitmap,bitmap.image_size(),cudaMemcpyDeviceToHost));


    bitmap.show_image();
    return 0;
}