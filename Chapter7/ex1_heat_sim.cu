#include "../common/book.h"
#include "../common/image.h"
#include "cuda_runtime.h"

#include <algorithm>

const int DIM = 1024;
const float SPEED = 0.25f;
const int ROUND = 90; //效果比较好的迭代次数
const float MAX_TEMP = 1.0f;
const float MIN_TEMP = 0.0001f;

struct DataBlock{
    unsigned char *output_bitmap;
    float *dev_inSrc;
    float *dev_outSrc;
    float *dev_constSrc;
    IMAGE *bitmap;

    cudaEvent_t start, stop;
    float totalTime;
    int frames;
};


void init(DataBlock *data,IMAGE *bitmap_ptr){
    data->bitmap = bitmap_ptr;
    data->totalTime = 0;
    data->frames = 0;

    HANDLE_ERROR(cudaEventCreate(&data->start));
    HANDLE_ERROR(cudaEventCreate(&data->stop));

    HANDLE_ERROR(cudaMalloc((void**)&data->output_bitmap,data->bitmap->image_size()));
    HANDLE_ERROR(cudaMalloc((void**)&data->dev_inSrc,data->bitmap->image_size()));
    HANDLE_ERROR(cudaMalloc((void**)&data->dev_outSrc,data->bitmap->image_size()));
    HANDLE_ERROR(cudaMalloc((void**)&data->dev_constSrc,data->bitmap->image_size()));

    float *temp = (float*)malloc(sizeof(float) * data->bitmap->image_size());
    for (int i = 0; i < DIM * DIM; i++) {
        temp[i] = 0;
        int x = i % DIM;
        int y = i / DIM;
        if ((x > 300) && (x < 600) && (y > 310) && (y < 601))
            temp[i] = MAX_TEMP;
    }
    temp[DIM * 100 + 100] = (MAX_TEMP + MIN_TEMP) / 2;
    temp[DIM * 700 + 100] = MIN_TEMP;
    temp[DIM * 300 + 300] = MIN_TEMP;
    temp[DIM * 200 + 700] = MIN_TEMP;
    for (int y = 800; y < 900; y++) {
        for (int x = 400; x < 500; x++) {
            temp[x + y * DIM] = MIN_TEMP;
        }
    }
    HANDLE_ERROR(cudaMemcpy(data->dev_constSrc,temp,data->bitmap->image_size(),cudaMemcpyHostToDevice));

    for (int y = 800; y < DIM; y++) {
        for (int x = 0; x < 200; x++) {
            temp[x + y * DIM] = MAX_TEMP;
        }
    }

    HANDLE_ERROR(cudaMemcpy(data->dev_inSrc,temp,data->bitmap->image_size(),cudaMemcpyHostToDevice));
    free(temp);
}

__global__ void copy_const_kernel(float *iptr,const float *cptr){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    if(cptr[offset] != 0){
        iptr[offset] = cptr[offset];
    }
}

__global__ void blend_kernel(float *outSrc,const float *inSrc,const float *constSrc){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    int left = offset - 1;
    int right = offset + 1;
    int top = offset - DIM;
    int bottom = offset + DIM;
    if(x == 0){
        left++;
    }
    if(x == DIM - 1){
        right--;
    }
    if(y == 0){
        top += DIM;
    }
    if(y == DIM - 1){
        bottom -= DIM;
    }

    outSrc[offset] = inSrc[offset] + SPEED * 
        (inSrc[top] + inSrc[bottom] + inSrc[left] + inSrc[right] - inSrc[offset] * 4);
}


void image_generate(DataBlock *data){
    HANDLE_ERROR(cudaEventRecord(data->start, 0));
    dim3 blocks(DIM / 16, DIM / 16);
    dim3 threads(16, 16);

    IMAGE* bitmap = data->bitmap;

    for(int i = 0;i < ROUND;i++){
        copy_const_kernel<<<blocks,threads>>>(data->dev_inSrc,data->dev_constSrc);
        blend_kernel<<<blocks,threads>>>(data->dev_outSrc,data->dev_inSrc,data->dev_constSrc);
        std::swap(data->dev_inSrc,data->dev_outSrc);
    }
    
    float_to_color<<<blocks, threads>>>(data->output_bitmap, data->dev_inSrc);

    HANDLE_ERROR(cudaEventRecord(data->stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(data->stop));
    float elapsedTime;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, data->start, data->stop));
    data->totalTime += elapsedTime;
    data->frames++;
    printf("Time to generate %d frame: %3.1f ms\n", data->frames, elapsedTime);

}

void clear(DataBlock *data){
    cudaFree(data->output_bitmap);
    cudaFree(data->dev_inSrc);
    cudaFree(data->dev_outSrc);
    cudaFree(data->dev_constSrc);

    cudaEventDestroy(data->start);
    cudaEventDestroy(data->stop);

    HANDLE_ERROR(cudaEventDestroy(data->start));
    HANDLE_ERROR(cudaEventDestroy(data->stop));
}

int main(){
    DataBlock data;
    IMAGE bitmap(DIM, DIM);

    init(&data, &bitmap);
    
    while(1){
        /*
            at frame 2500:17.5ms preframe
        */
        image_generate(&data);
        HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), data.output_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost));
        bitmap.show_image(30);
    }


    clear(&data);

    return 0;
}


