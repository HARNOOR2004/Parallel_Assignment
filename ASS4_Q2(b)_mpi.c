
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#define N 1000

void merge(int *arr, int l, int m, int r, int *temp) {
    int i = l, j = m + 1, k = l;
    while (i <= m && j <= r) {
        if (arr[i] < arr[j]) temp[k++] = arr[i++];
        else temp[k++] = arr[j++];
    }
    while (i <= m) temp[k++] = arr[i++];
    while (j <= r) temp[k++] = arr[j++];
    for (i = l; i <= r; i++) arr[i] = temp[i];
}

void mergeSortParallel(int *arr) {
    int *temp = (int *)malloc(N * sizeof(int));
    for (int width = 1; width < N; width *= 2) {
        #pragma omp parallel for
        for (int i = 0; i < N; i += 2 * width) {
            int l = i;
            int m = i + width - 1;
            int r = i + 2 * width - 1;
            if (r >= N) r = N - 1;
            if (m < r) merge(arr, l, m, r, temp);
        }
    }
    free(temp);
}
