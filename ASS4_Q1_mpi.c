#include <stdio.h>
#include <cuda.h>

#define N 1024

// CUDA Kernel
__global__ void sumKernel(int* input, int* output) {
    int tid = threadIdx.x;

    if (tid == 0) {
        // Task A: Iterative sum
        int sum = 0;
        for (int i = 0; i < N; i++) {
            sum += input[i];
        }
        output[0] = sum;
    }
    else if (tid == 1) {
        // Task B: Direct formula sum
        int sum = N * (N - 1) / 2; 
        output[1] = sum;
    }
}

int main() {
    int *h_input, *h_output;
    int *d_input, *d_output;

    
    h_input = (int*)malloc(N * sizeof(int));
    h_output = (int*)malloc(2 * sizeof(int)); 

    
    for (int i = 0; i < N; i++) {
        h_input[i] = i;
    }

    
    cudaMalloc((void**)&d_input, N * sizeof(int));
    cudaMalloc((void**)&d_output, 2 * sizeof(int));

    
    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);


    sumKernel<<<1, 2>>>(d_input, d_output);

   
    cudaMemcpy(h_output, d_output, 2 * sizeof(int), cudaMemcpyDeviceToHost);

   
    printf("Iterative Sum (Thread 0): %d\n", h_output[0]);
    printf("Formula Sum (Thread 1): %d\n", h_output[1]);

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
