#include<stdio.h>
#include<cuda_runtime.h>

const int m = 2, n = 3;

__global__ void mat_add_kernel(int a[m][n], int b[m][n], int c[m][n]) {
    int i = threadIdx.x;
    int j = threadIdx.y;
    if(i < m && j < n) c[i][j] = a[i][j] + b[i][j];
}

int main() {
    int h_a[m][n] = {{1, 2, 3}, {4, 5, 6}};
    int h_b[m][n] = {{10, 10, 10}, {10, 10, 10}};
    int (*d_a)[n], (*d_b)[n], (*d_c)[n];

    cudaError_t err;

    err = cudaMalloc(&d_a, m * n * sizeof(int));
    if (err != cudaSuccess) printf("cudaMalloc d_a failed\n");
    err = cudaMemcpy(d_a, h_a, m * n * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) printf("cudaMemcpy h_a -> d_a failed: %s\n", cudaGetErrorString(err));
    err = cudaMalloc(&d_b, m * n * sizeof(int));
    if (err != cudaSuccess) printf("cudaMalloc d_b failed\n");
    err = cudaMemcpy(d_b, h_b, m * n * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) printf("cudaMemcpy h_b -> d_b failed: %s\n", cudaGetErrorString(err));
    err = cudaMalloc(&d_c, m * n * sizeof(int));
    if (err != cudaSuccess) printf("cudaMalloc d_c failed\n");

    int blks = 1;
    dim3 threads(m, n);
    mat_add_kernel<<<blks, threads>>>(d_a, d_b, d_c);
    err = cudaGetLastError();
    if (err != cudaSuccess) printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    cudaDeviceSynchronize();

    int h_c[m][n];
    err = cudaMemcpy(h_c, d_c, m * n * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) printf("cudaMemcpy d_c -> h_c failed: %s\n", cudaGetErrorString(err));

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) printf("%d\t", h_c[i][j]);
        printf("\n");
    }
    printf("\n\n");

    return 0; 
}