#include "cuda.h"
#include "../common/book.h"
#include "../common/image.h"
/* 
	光线追踪
	根据光路可逆的原理，
	我们可以通过从像素反向追踪光线，来确定光线的路径
*/

const double INF = 2e10;
const int DIM = 1024;
const int SPHERES = 200;
#define rnd(x) (x * rand() / RAND_MAX)


struct Sphere{
	double r, b, g;
	double radius;
	double x, y, z;

	__device__ double hit(double ox, double oy, double *n){
		double dx = ox - x;
		double dy = oy - y;
		if(dx * dx + dy * dy < radius * radius){
			double dz = sqrtf(radius * radius - dx * dx - dy * dy);
			*n = dz / sqrtf(radius * radius);
			return z + dz;
			/*
				感觉这里摄像机的位置决定了应该是z + dz还是z - dz
				z + dz是摄像机向z轴负方向看过去的
				z - dz是摄像机从z轴正方向看过去的
			*/
		}
		return -INF;
	}
};

// Sphere* dev_sphere; // 498.3ms
__constant__ Sphere dev_sphere[SPHERES]; // 87.9ms

/*
	常量内存静态地分配空间
	不需要cudaMalloc，也不需要cudaFree

	注意常量内存申明后应该直接调用而非传参访问，否则会出现错误
	传参访问的指针仍然指向全局内存，而不是常量内存

	常量内存把访问限制在了只读，但是它的访问速度比全局内存快
	单次读取可以广播到NearBy Threads
	对连续的内存访问有很好的性能

	线程束是指包含32个线程的集合，
	每次访问常量内存时，数据会被广播到线程束中

	但是如果线程束中的线程访问的地址不同，
	会造成负优化，不同的读取操作会串行化，
	而全局内存的读取操作是并行的。
*/

__global__ void kernel(unsigned char *ptr){
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;
	
	double ox = (x - DIM / 2);
	double oy = (y - DIM / 2);

	double r = 0, g = 0, b = 0;
	double maxz = -INF;
	for(int i = 0;i < SPHERES;i++){
		double n;
		double t = dev_sphere[i].hit(ox, oy, &n);
		if(t > maxz){
			double fscale = n;
			r = dev_sphere[i].r * fscale;
			g = dev_sphere[i].g * fscale;
			b = dev_sphere[i].b * fscale;
			maxz = t;
		}
	}
	ptr[offset * 4 + 0] = (int)(r * 255);
	ptr[offset * 4 + 1] = (int)(g * 255);
	ptr[offset * 4 + 2] = (int)(b * 255);
	ptr[offset * 4 + 3] = 255;
}

int main(){

	cudaEvent_t start, stop;
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
	HANDLE_ERROR(cudaEventRecord(start, 0));
	/*
		计算程序运行时间
		创建一个事件，记录开始时间

		但CPU和GPU是异步的，所以需要cudaEventSynchronize(stop)来同步执行
	*/

	IMAGE bitmap(DIM, DIM);
	unsigned char *dev_bitmap;

	HANDLE_ERROR(cudaMalloc((void **)&dev_bitmap, bitmap.image_size()));
	// HANDLE_ERROR(cudaMalloc((void **)&dev_sphere, SPHERES * sizeof(Sphere)));

	Sphere *temp_s = (Sphere *)malloc(sizeof(Sphere) * SPHERES);
	for(int i = 0;i < SPHERES;i++){
		temp_s[i] = Sphere{rnd(1.0),rnd(1.0),rnd(1.0),rnd(100.0) + 20,rnd(1024.0) - 512,rnd(1024.0) - 512,rnd(1024.0) - 512};
	}
	// HANDLE_ERROR(cudaMemcpy(dev_sphere, temp_s, SPHERES * sizeof(Sphere), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToSymbol(dev_sphere, temp_s, SPHERES * sizeof(Sphere)));
	/*
		复制常量内存用cudaMemcpyToSymbol
	*/
	free(temp_s);

	dim3 blocks(DIM / 16, DIM / 16);
	dim3 threads(16, 16);
	// kernel<<<blocks, threads>>>(dev_bitmap,dev_sphere);
	kernel<<<blocks, threads>>>(dev_bitmap);
	
	HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost));
	
	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));
	/*
	cudaEventRecord(event) 记录事件，
	但不保证当它返回时，事件已经完成。
	当之前的操作完成后，这个事件才会被认为是触发的。
	cudaEventSynchronize(event) 用于主机线程阻塞，等待事件触发。
	主机线程会暂停，直到与这个事件相关的所有GPU操作都执行完毕。
	*/
	float elapsedTime;
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
	printf("Time to generate: %3.1f ms\n", elapsedTime);
	HANDLE_ERROR(cudaEventDestroy(start));
	HANDLE_ERROR(cudaEventDestroy(stop));

	
	bitmap.show_image();
	// bitmap.save_image("ray.jpg");


	cudaFree(dev_bitmap);
	// cudaFree(dev_sphere);
	return 0;
}