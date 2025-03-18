#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 100  // Size of array

int main(int argc, char **argv) {
    int rank, size;
    int arr[N], local_sum = 0, global_sum = 0;
    int local_n;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    local_n = N / size;
    int local_arr[local_n];

    if (rank == 0) {
        srand(42);
        for (int i = 0; i < N; i++) {
            arr[i] = rand() % 10;
        }
        printf("Array: ");
        for (int i = 0; i < N; i++) {
            printf("%d ", arr[i]);
        }
        printf("\n");
    }

    MPI_Scatter(arr, local_n, MPI_INT, local_arr, local_n, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < local_n; i++) {
        local_sum += local_arr[i];
    }

    MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Global sum: %d\n", global_sum);
    }

    MPI_Finalize();
    return 0;
}