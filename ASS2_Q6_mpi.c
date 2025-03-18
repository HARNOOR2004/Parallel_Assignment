#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 100  // Vector size

int main(int argc, char **argv) {
    int rank, size;
    double A[N], B[N], local_dot = 0.0, global_dot = 0.0;
    int local_n;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    local_n = N / size;
    double local_A[local_n], local_B[local_n];

    if (rank == 0) {
        srand(42);
        for (int i = 0; i < N; i++) {
            A[i] = rand() % 10;
            B[i] = rand() % 10;
        }
        printf("Vector A: ");
        for (int i = 0; i < N; i++) {
            printf("%.0f ", A[i]);
        }
        printf("\nVector B: ");
        for (int i = 0; i < N; i++) {
            printf("%.0f ", B[i]);
        }
        printf("\n");
    }

    MPI_Scatter(A, local_n, MPI_DOUBLE, local_A, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(B, local_n, MPI_DOUBLE, local_B, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < local_n; i++) {
        local_dot += local_A[i] * local_B[i];
    }

    MPI_Reduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Dot Product: %.2f\n", global_dot);
    }

    MPI_Finalize();
    return 0;
}