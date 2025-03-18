#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 20  // Size of array

void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

void odd_even_sort(int *local_arr, int local_n, int rank, int size) {
    int phase, partner;
    for (phase = 0; phase < size; phase++) {
        if (phase % 2 == 0) {
            partner = (rank % 2 == 0) ? rank + 1 : rank - 1;
        } else {
            partner = (rank % 2 == 0) ? rank - 1 : rank + 1;
        }

        if (partner >= 0 && partner < size) {
            int recv_arr[local_n];
            MPI_Sendrecv(local_arr, local_n, MPI_INT, partner, 0,
                         recv_arr, local_n, MPI_INT, partner, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (rank < partner) {
                for (int i = 0; i < local_n; i++) {
                    if (local_arr[i] > recv_arr[i]) {
                        swap(&local_arr[i], &recv_arr[i]);
                    }
                }
            } else {
                for (int i = 0; i < local_n; i++) {
                    if (local_arr[i] < recv_arr[i]) {
                        swap(&local_arr[i], &recv_arr[i]);
                    }
                }
            }
        }
    }
}

int main(int argc, char **argv) {
    int rank, size;
    int arr[N];
    int local_n;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    local_n = N / size;
    int local_arr[local_n];

    if (rank == 0) {
        srand(42);
        for (int i = 0; i < N; i++) {
            arr[i] = rand() % 100;
        }
        printf("Unsorted array: ");
        for (int i = 0; i < N; i++) {
            printf("%d ", arr[i]);
        }
        printf("\n");
    }

    MPI_Scatter(arr, local_n, MPI_INT, local_arr, local_n, MPI_INT, 0, MPI_COMM_WORLD);

    // Local sort
    for (int i = 0; i < local_n - 1; i++) {
        for (int j = 0; j < local_n - i - 1; j++) {
            if (local_arr[j] > local_arr[j + 1]) {
                swap(&local_arr[j], &local_arr[j + 1]);
            }
        }
    }

    // Odd-Even Sort
    odd_even_sort(local_arr, local_n, rank, size);

    MPI_Gather(local_arr, local_n, MPI_INT, arr, local_n, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Sorted array: ");
        for (int i = 0; i < N; i++) {
            printf("%d ", arr[i]);
        }
        printf("\n");
    }

    MPI_Finalize();
    return 0;
}