#include<stdio.h>
#include<cuda_runtime.h>
const int n = 4;

__global__ void vec_add(int* a, int* b, int* c) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main() {
    int h_a[n] = {1, 2, 3, 4}, h_b[n] = {10, 20, 30,  40};
    int *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, n * sizeof(int));
    cudaMemcpy(d_a, h_a, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&d_b, n * sizeof(int));
    cudaMemcpy(d_b, h_b, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&d_c, n * sizeof(int));

    int blk = 1, threads = n;
    vec_add<<<blk, threads>>>(d_a, d_b, d_c);
    cudaDeviceSynchronize();

    int h_c[n];
    cudaMemcpy(h_c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);

    for(int i = 0; i < n; i++) printf("%d ", h_c[i]);
    return 0; 
}