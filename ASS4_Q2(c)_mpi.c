
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

extern void mergeSortParallel(int *arr);
extern void cudaMergeSort(int *arr);

#define N 1000

void fillArray(int *arr) {
    for (int i = 0; i < N; i++)
        arr[i] = rand() % 10000;
}

int main() {
    int arr1[N], arr2[N];
    fillArray(arr1);
    for (int i = 0; i < N; i++) arr2[i] = arr1[i];

    clock_t start, end;

    start = clock();
    mergeSortParallel(arr1);
    end = clock();
    printf("CPU Pipelined Merge Sort Time: %lf sec\n", (double)(end - start) / CLOCKS_PER_SEC);

    start = clock();
    cudaMergeSort(arr2);
    end = clock();
    printf("CUDA Merge Sort Time: %lf sec\n", (double)(end - start) / CLOCKS_PER_SEC);

    return 0;
}
