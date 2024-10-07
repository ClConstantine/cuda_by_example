#include "../common/book.h"
#include "../common/image.h"
#include "cuda_runtime.h"

#include <algorithm>

/*
    纹理内存 Texture Memory
    纹理内存也是一种只读内存，
    在访问大量空间局部性的数据中有很好的性能。
    但这种情况较少

    注意，Texture reference 在12.0被彻底弃用
    CUDA 12.0之后需要考虑使用Texture Object
*/

const int DIM = 1024;
const float SPEED = 0.25f;
const int ROUND = 90; //效果比较好的迭代次数
const float MAX_TEMP = 1.0f;
const float MIN_TEMP = 0.0001f;

texture<float,2> texConstSrc;
texture<float,2> texIn;
texture<float,2> texOut;

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

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    /* 
       2D纹理绑定
       offset，texture，pointer，格式描述符desc，数据指针，描述符，宽度，高度，每行的字节数
    */
    HANDLE_ERROR(cudaBindTexture2D(0,texConstSrc,data->dev_constSrc,desc,DIM,DIM,sizeof(float) * DIM));
    HANDLE_ERROR(cudaBindTexture2D(0,texIn,data->dev_inSrc,desc,DIM,DIM,sizeof(float) * DIM));
    HANDLE_ERROR(cudaBindTexture2D(0,texOut,data->dev_outSrc,desc,DIM,DIM,sizeof(float) * DIM));
    
    // HANDLE_ERROR(cudaBindTexture(0,texConstSrc,data->dev_constSrc,data->bitmap->image_size()));
    // HANDLE_ERROR(cudaBindTexture(0,texIn,data->dev_inSrc,data->bitmap->image_size()));
    // HANDLE_ERROR(cudaBindTexture(0,texOut,data->dev_outSrc,data->bitmap->image_size()));
    
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

__global__ void copy_const_kernel(float *iptr){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float c = tex2D(texConstSrc,x,y);

    if(c != 0){
        iptr[offset] = c;
    }
}

__global__ void blend_kernel(float *dst,bool dstOut){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float t,l,c,r,b;

    /*
        自动处理越界的访问
    */
    if(dstOut){
        t = tex2D(texIn,x,y-1);
        l = tex2D(texIn,x-1,y);
        c = tex2D(texIn,x,y);
        r = tex2D(texIn,x+1,y);
        b = tex2D(texIn,x,y+1);
    }
    else{
        t = tex2D(texOut,x,y-1);
        l = tex2D(texOut,x-1,y);
        c = tex2D(texOut,x,y);
        r = tex2D(texOut,x+1,y);
        b = tex2D(texOut,x,y+1);
    }
    dst[offset] = c + SPEED * (t + b + r + l - 4 * c);

}

void image_generate(DataBlock *data){
    HANDLE_ERROR(cudaEventRecord(data->start, 0));
    dim3 blocks(DIM / 16, DIM / 16);
    dim3 threads(16, 16);

    IMAGE* bitmap = data->bitmap;

    volatile bool dstOut = true;

    for(int i = 0;i < ROUND;i++){
        float *in, *out;
        if(dstOut){
            in = data->dev_inSrc;
            out = data->dev_outSrc;
        }
        else{
            in = data->dev_outSrc;
            out = data->dev_inSrc;
        }

        copy_const_kernel<<<blocks,threads>>>(in);
        blend_kernel<<<blocks,threads>>>(out,dstOut);
        dstOut = !dstOut;
    }
    
    float_to_color<<<blocks, threads>>>(data->output_bitmap, data->dev_outSrc);

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

    cudaUnbindTexture(texConstSrc);
    cudaUnbindTexture(texIn);
    cudaUnbindTexture(texOut);

}

int main(){
    DataBlock data;
    IMAGE bitmap(DIM, DIM);

    init(&data, &bitmap);
    
    while(1){
        /*
            at frame 2500:18.5ms preframe
            really useful?
        */
        image_generate(&data);
        HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), data.output_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost));
        bitmap.show_image(30);
    }


    clear(&data);

    return 0;
}


