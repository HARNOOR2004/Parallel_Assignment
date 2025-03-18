#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 16  // Array size

int main(int argc, char **argv) {
    int rank, size;
    int arr[N], local_sum = 0, prefix_sum = 0;
    int local_n;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    local_n = N / size;
    int local_arr[local_n], local_prefix[local_n];

    if (rank == 0) {
        srand(42);
        for (int i = 0; i < N; i++) {
            arr[i] = rand() % 10;
        }
        printf("Original array: ");
        for (int i = 0; i < N; i++) {
            printf("%d ", arr[i]);
        }
        printf("\n");
    }

    MPI_Scatter(arr, local_n, MPI_INT, local_arr, local_n, MPI_INT, 0, MPI_COMM_WORLD);

    local_prefix[0] = local_arr[0];
    for (int i = 1; i < local_n; i++) {
        local_prefix[i] = local_prefix[i - 1] + local_arr[i];
    }
    
    MPI_Exscan(&local_prefix[local_n - 1], &prefix_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    if (rank != 0) {
        for (int i = 0; i < local_n; i++) {
            local_prefix[i] += prefix_sum;
        }
    }
    
    MPI_Gather(local_prefix, local_n, MPI_INT, arr, local_n, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Prefix sum: ");
        for (int i = 0; i < N; i++) {
            printf("%d ", arr[i]);
        }
        printf("\n");
    }

    MPI_Finalize();
    return 0;
}