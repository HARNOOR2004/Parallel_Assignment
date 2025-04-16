# ASS3_Q3_cuda.cu - Threads Performing Different Tasks (CUDA)

## Description:

A CUDA-based program where multiple threads perform different tasks:
- **Thread 0**: Computes the sum of the first N integers using an iterative approach.
- **Thread 1**: Computes the sum using the direct mathematical formula.

## Steps Performed:
1. Define the value of N (1024).
2. Allocate memory for result storage on the host and device.
3. Launch kernel with at least two threads (each doing a distinct task).
4. Copy results from device to host and display the outputs.

## Compilation and Execution

```sh
nvcc ASS4_Q1_cuda.cu -o task_threads
./task_threads
```