#include <iostream>
#include <assert.h>
#include <cstdlib>
#include <time.h>
#include <numeric>
#include "../common/book.h"
#include "lock.h"

const int N = 1024 * 1024 * 50;
const int ELEMENTS = N / sizeof(int);
const int BASE = 1024;

struct node{
	int key;
	int value;
	node *next;
};

struct HashTable{
	int size;
	node **entries;
	node *pool;
};

void init_table(HashTable &table, int entries, int elements){
	table.size = entries;

	cudaMalloc((void**)&table.entries, sizeof(node *) * entries);
	cudaMemset(table.entries, 0, sizeof(node *) * entries);
	cudaMalloc((void**)&table.pool, sizeof(node) * elements);

}

void free_table(HashTable &table){
	cudaFree(table.entries);
	cudaFree(table.pool);
}

void copy_table_to_host(HashTable &hostTable, HashTable &deviceTable){
	hostTable.size = deviceTable.size;
	hostTable.entries = (node**)malloc(sizeof(node*) * hostTable.size);
	hostTable.pool = (node*)malloc(sizeof(node) * ELEMENTS);

	cudaMemcpy(hostTable.entries, deviceTable.entries, sizeof(node*) * hostTable.size, cudaMemcpyDeviceToHost);
	cudaMemcpy(hostTable.pool, deviceTable.pool, sizeof(node) * ELEMENTS, cudaMemcpyDeviceToHost);

	for(int i = 0;i < hostTable.size; i++){
		/*
			注意这里不能直接访问deviceTable.entries[i]
			因为deviceTable.entries[i]分配在device上，host无法访问
			由于hostTable是deviceTable的拷贝，
			所以访问hostTable.entries[i]相当于访问deviceTable.entries[i]
		*/
		if(hostTable.entries[i] != NULL){
			hostTable.entries[i] = hostTable.pool + (hostTable.entries[i] - deviceTable.pool);
		}
	}

	for(int i = 0;i < ELEMENTS; i++){
		if(hostTable.pool[i].next != NULL){
			hostTable.pool[i].next = hostTable.pool + (hostTable.pool[i].next - deviceTable.pool);
		}
	}
}

__device__ __host__ int hash(int key, int size){
	return (key % size + size) % size;
}

/*
	注意在host上管理一个class时，传入的是拷贝！
	如果是引用，device无法访问host的内存，就会出错
*/

__global__ void add_to_table(Lock *lock, HashTable Table, int *keys, int *values){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	while(index < ELEMENTS){
		int key = keys[index];
		int hashValue = hash(key, BASE);
		
		for(int i = 0; i < 32; i++){
			if(index % 32 == i){
				node *newNode = &(Table.pool[index]);
				newNode->key = key;
				newNode->value = values[index];
				
				lock[hashValue].lock();
				newNode->next = Table.entries[hashValue];
				Table.entries[hashValue] = newNode;
				lock[hashValue].unlock();
			}
		}
		index += stride;
	}

}

void verify(HashTable &deviceTable){
	HashTable hostTable;
	copy_table_to_host(hostTable, deviceTable);
	
	int count = 0;
	for(int i = 0; i < hostTable.size; i++){
		node *cur = hostTable.entries[i];
		while(cur != NULL){
			count++;
			assert(hash(cur->key, hostTable.size) == i);
			cur = cur->next;
		}
	}
	assert(count == ELEMENTS);

	free(hostTable.entries);
	free(hostTable.pool);
}

int main(){

	int *buffer = (int*)big_random_block(N);
	int *values = (int*)malloc(N);
	std::iota(values, values + N/sizeof(int), 0);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	int *dev_keys, *dev_values;
	cudaMalloc((void**)&dev_keys, N);
	cudaMalloc((void**)&dev_values, N);
	cudaMemcpy(dev_keys, buffer, N, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_values, values, N, cudaMemcpyHostToDevice);	

	HashTable table;
	init_table(table,BASE,ELEMENTS);

	Lock lock[BASE];
	Lock *dev_lock;
	cudaMalloc((void**)&dev_lock, sizeof(Lock) * BASE);
	cudaMemcpy(dev_lock, lock, sizeof(Lock) * BASE, cudaMemcpyHostToDevice);

	add_to_table<<<128,256>>>(dev_lock, table, dev_keys, dev_values);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsed_time;	
	cudaEventElapsedTime(&elapsed_time, start, stop);
	printf("Time cost: %f ms\n", elapsed_time);

	verify(table);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	free_table(table);

	free(buffer);
	free(values);
	cudaFree(dev_keys);
	cudaFree(dev_values);
	cudaFree(dev_lock);

	return 0;
}