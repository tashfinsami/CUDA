#include<stdio.h>
#include<cuda_runtime.h>
const int n = 4;

__global__ void vec_add_kernel(int* a, int* b, int* c) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main() {
    int h_a[n] = {1, 2, 3, 4}, h_b[n] = {10, 20, 30,  40};
    int *d_a, *d_b, *d_c;

    cudaError_t err;
    err = cudaMalloc(&d_a, n * sizeof(int));
    if (err != cudaSuccess) printf("cudaMalloc d_a failed\n");
    err = cudaMemcpy(d_a, h_a, n * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) printf("cudaMemcpy h_a -> d_a failed: %s\n", cudaGetErrorString(err));
    err = cudaMalloc(&d_b, n * sizeof(int));
    if (err != cudaSuccess) printf("cudaMalloc d_b failed\n");
    err = cudaMemcpy(d_b, h_b, n * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) printf("cudaMemcpy h_b -> d_b failed: %s\n", cudaGetErrorString(err));
    err = cudaMalloc(&d_c, n * sizeof(int));
    if (err != cudaSuccess) printf("cudaMalloc d_c failed\n");

    int blk = 1, threads = n;
    vec_add_kernel<<<blk, threads>>>(d_a, d_b, d_c);
    err = cudaGetLastError();
    if (err != cudaSuccess) printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    cudaDeviceSynchronize();

    int h_c[n];
    err = cudaMemcpy(h_c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) printf("cudaMemcpy d_c -> h_c failed: %s\n", cudaGetErrorString(err));

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    for(int i = 0; i < n; i++) printf("%d ", h_c[i]);
    printf("\n");

    return 0; 
}