#include <stdio.h>
#include <math.h>

#define N 1024

__device__ float inputA[N];
__device__ float output[N];

__global__ void vectorSqrt() {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < N) {
        output[idx] = sqrtf(inputA[idx]);
    }
}

int main() {
    float hostInputA[N], hostOutput[N];

    
    for (int i = 0; i < N; ++i) {
        hostInputA[i] = (float)(i + 1);
    }

    cudaMemcpyToSymbol(inputA, hostInputA, N * sizeof(float));


    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorSqrt<<<blocksPerGrid, threadsPerBlock>>>();


    cudaMemcpyFromSymbol(hostOutput, output, N * sizeof(float));

   
    for (int i = 0; i < 10; ++i) {
        printf("sqrt(%f) = %f\n", hostInputA[i], hostOutput[i]);
    }

    return 0;
}
