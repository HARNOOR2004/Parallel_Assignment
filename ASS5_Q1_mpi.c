
#include <stdio.h>
#include <cuda.h>

#define N 1024 * 1024  
#define THREADS_PER_BLOCK 256

__device__ float d_A[N];
__device__ float d_B[N];
__device__ float d_C[N];

__global__ void vectorAdd() {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        d_C[i] = d_A[i] + d_B[i];
}

int main() {
    float h_A[N], h_B[N], h_C[N];


    for (int i = 0; i < N; i++) {
        h_A[i] = i * 1.0f;
        h_B[i] = i * 2.0f;
    }


    cudaMemcpyToSymbol(d_A, h_A, N * sizeof(float));
    cudaMemcpyToSymbol(d_B, h_B, N * sizeof(float));


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

 
    cudaEventRecord(start);


    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    vectorAdd<<<blocks, THREADS_PER_BLOCK>>>();


    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

  
    cudaMemcpyFromSymbol(h_C, d_C, N * sizeof(float));

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Kernel execution time: %f ms\n", ms);


    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    float memClock = prop.memoryClockRate * 1e3; 
    float memBusWidth = prop.memoryBusWidth;     
    float theoreticalBW = 2.0f * memClock * memBusWidth / 8.0f / 1e9; 

    printf("Theoretical Memory Bandwidth: %.2f GB/s\n", theoreticalBW);

 
    float totalBytes = 2 * N * sizeof(float) + N * sizeof(float); 
    float seconds = ms / 1000.0f;
    float measuredBW = totalBytes / seconds / 1e9; 

    printf("Measured Memory Bandwidth: %.2f GB/s\n", measuredBW);

  
    for (int i = 0; i < 5; i++) {
        printf("C[%d] = %f\n", i, h_C[i]);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
