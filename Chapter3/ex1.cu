#include <stdio.h>

__global__ void kernel() {
    
}

int main() {
    kernel<<<1, 1>>>();
    /*
        CUDA C需要通过某种语法将函数标记为在设备上执行，
        作为Device Code，这种语法是三个尖括号<<< >>>。
        这种语法被称为kernel调用，kernel是在设备上执行的函数。
    */
    printf("Hello, World!\n");

    return 0;
}