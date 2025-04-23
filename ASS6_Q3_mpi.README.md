# sqrt_vector_cuda.cu - CUDA Vector Square Root Benchmark

## Description:
This CUDA program computes the square root of each element in an input vector and stores the result in an output vector. The program has been enhanced to measure kernel execution time using CUDA events and benchmark performance over different input sizes.

## Benchmark Sizes:
- 50,000 elements
- 500,000 elements
- 5,000,000 elements
- 50,000,000 elements

## Steps:
1. Allocate memory for `N` elements on both host and device.
2. Initialize input data.
3. Launch CUDA kernel for computing square root element-wise.
4. Record the time taken by the kernel.
5. Copy and verify the results.
6. Repeat for each benchmark size.

## Compilation And Execution:
```sh
nvcc sqrt_vector_cuda.cu -o sqrt_vector
./sqrt_vector
```