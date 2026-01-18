#include <stdio.h>
#include <cuda_runtime.h>

__global__ void hello_kernel() {
    printf("TEST run from GPU thread %d\n", threadIdx.x);
}

int main() {
    hello_kernel<<<1, 378>>>();
    cudaDeviceSynchronize();
    return 0;
}
