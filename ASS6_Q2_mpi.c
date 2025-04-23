#include <stdio.h>
#include <math.h>

#define N 1024

__global__ void sqrtKernel(float *A, float *C) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < N) {
        C[idx] = sqrtf(A[idx]);
    }
}

int main() {
    float A[N], C[N];
    float *d_A, *d_C;

    for (int i = 0; i < N; i++) {
        A[i] = (float)(i + 1);
    }


    cudaMalloc((void**)&d_A, N * sizeof(float));
    cudaMalloc((void**)&d_C, N * sizeof(float));

    cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    sqrtKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_C);

    cudaMemcpy(C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

   
    for (int i = 0; i < 10; i++) {
        printf("sqrt(%f) = %f\n", A[i], C[i]);
    }

    cudaFree(d_A);
    cudaFree(d_C);

    return 0;
}