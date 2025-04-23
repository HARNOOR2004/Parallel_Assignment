# vector_sqrt_cuda.cu - CUDA Vector Square Root Computation

## Description:
Implements square root computation on a vector using CUDA. Each thread computes the square root of a single element from the input array using `sqrtf()`. This program highlights that square root operations are more computationally expensive than addition or multiplication.

## Steps:
1. Define an array of 1024 floats.
2. Use `__device__` global memory to allocate space on GPU.
3. Copy input data to device using `cudaMemcpyToSymbol`.
4. Launch a CUDA kernel with multiple threads.
5. Each thread computes the square root of one element.
6. Copy the result back using `cudaMemcpyFromSymbol`.

## Compilation And Execution:
```sh
nvcc vector_sqrt_cuda.cu -o vector_sqrt
./vector_sqrt
```