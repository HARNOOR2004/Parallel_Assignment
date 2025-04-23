# sqrt_vector_cuda.cu - CUDA Vector Square Root Calculation

## Description:
A CUDA program that computes the square root of each element in an input vector `A[i]` and stores the results in the output vector `C[i]`. Each thread handles the computation for one element, showcasing parallel element-wise operations using GPU acceleration.

## Steps:
1. Define and initialize an array `A` of size 1024.
2. Allocate memory on the device for both input and output arrays.
3. Copy the input array `A` to device memory.
4. Launch a CUDA kernel where each thread computes `C[i] = sqrt(A[i])`.
5. Copy the result array `C` back from the device to the host.
6. Print the first few results for verification.

## Compilation And Execution:
```sh
nvcc sqrt_vector_cuda.cu -o sqrt_vector
./sqrt_vector
```