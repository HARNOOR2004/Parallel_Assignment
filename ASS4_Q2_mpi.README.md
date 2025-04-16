# ASS3_Q4 - Parallel Merge Sort Using CPU Pipelining and CUDA

## Description:

This assignment implements merge sort in two parallel approaches:
1. **CPU Pipelining**: Using OpenMP to parallelize the merge operations.
2. **CUDA Merge Sort**: Each thread handles parts of the array in a hierarchical fashion.

The two methods are benchmarked for performance comparison on an array of size N=1000.

---

## Files:
- `ASS4_Q2(a)_cpu_pipelined.c`: OpenMP-based pipelined merge sort.
- `ASS4_Q2(b)_cuda_mergesort.cu`: CUDA-based merge sort kernel and implementation.
- `ASS4_Q2(c)_main.c`: Main function to test and compare both implementations.

---

## Compilation:

### CPU Pipelining (OpenMP)
```sh
gcc ASS4_Q2_main.c ASS4_Q2_cpu_pipelined.c -fopenmp -o cpu_sort
```