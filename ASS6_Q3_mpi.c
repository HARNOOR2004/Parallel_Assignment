#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#define NUM_SIZES 4
const int sizes[NUM_SIZES] = {50000, 500000, 5000000, 50000000};

__global__ void sqrtKernel(float *A, float *C, int N) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < N) {
        C[idx] = sqrtf(A[idx]);
    }
}

int main() {
    for (int s = 0; s < NUM_SIZES; s++) {
        int N = sizes[s];
        float *A, *C;
        float *d_A, *d_C;
        size_t size = N * sizeof(float);

        A = (float*)malloc(size);
        C = (float*)malloc(size);

        for (int i = 0; i < N; i++) {
            A[i] = (float)(i + 1);
        }

        cudaMalloc((void**)&d_A, size);
        cudaMalloc((void**)&d_C, size);
        cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        int threadsPerBlock = 256;
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        sqrtKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_C, N);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        printf("Time taken for square root computation with %d elements: %f ms\n", N, milliseconds);

        cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

        printf("First 5 results for N = %d:\n", N);
        for (int i = 0; i < 5; i++) {
            printf("sqrt(%f) = %f\n", A[i], C[i]);
        }

        cudaFree(d_A);
        cudaFree(d_C);
        free(A);
        free(C);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        printf("-----------------------------\n");
    }
    return 0;
}
