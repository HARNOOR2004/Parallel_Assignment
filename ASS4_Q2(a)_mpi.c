
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

#define N 1000

__device__ void merge(int *arr, int l, int m, int r, int *temp) {
    int i = l, j = m + 1, k = l;
    while (i <= m && j <= r) {
        if (arr[i] < arr[j]) temp[k++] = arr[i++];
        else temp[k++] = arr[j++];
    }
    while (i <= m) temp[k++] = arr[i++];
    while (j <= r) temp[k++] = arr[j++];
    for (i = l; i <= r; i++) arr[i] = temp[i];
}

__global__ void mergeSortKernel(int *arr, int *temp, int n, int width) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int start = tid * (2 * width);
    if (start + 2 * width - 1 < n) {
        int mid = start + width - 1;
        int end = start + 2 * width - 1;
        merge(arr, start, mid, end, temp);
    }
}

void cudaMergeSort(int *arr) {
    int *d_arr, *d_temp;
    cudaMalloc(&d_arr, N * sizeof(int));
    cudaMalloc(&d_temp, N * sizeof(int));
    cudaMemcpy(d_arr, arr, N * sizeof(int), cudaMemcpyHostToDevice);

    for (int width = 1; width < N; width *= 2) {
        int numThreads = (N + 2 * width - 1) / (2 * width);
        mergeSortKernel<<<(numThreads + 255) / 256, 256>>>(d_arr, d_temp, N, width);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
    cudaFree(d_temp);
}
