#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

#define N 70

void matrix_multiply_serial(double A[N][N], double B[N][N], double C[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0.0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void matrix_multiply_parallel(double A[N][N], double B[N][N], double C[N][N], int rank, int size) {
    int rows_per_proc = N / size;
    double local_C[rows_per_proc][N];

    for (int i = 0; i < rows_per_proc; i++) {
        for (int j = 0; j < N; j++) {
            local_C[i][j] = 0.0;
            for (int k = 0; k < N; k++) {
                local_C[i][j] += A[rank * rows_per_proc + i][k] * B[k][j];
            }
        }
    }

    MPI_Gather(local_C, rows_per_proc * N, MPI_DOUBLE, C, rows_per_proc * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

int main(int argc, char** argv) {
    int rank, size;
    double A[N][N], B[N][N], C[N][N];
    double start_time, run_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i][j] = rand() % 10;
                B[i][j] = rand() % 10;
            }
        }
    }

    MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(A, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        start_time = omp_get_wtime();
        matrix_multiply_serial(A, B, C);
        run_time = omp_get_wtime() - start_time;
        printf("Serial execution time: %f seconds\n", run_time);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = omp_get_wtime();
    matrix_multiply_parallel(A, B, C, rank, size);
    run_time = omp_get_wtime() - start_time;

    if (rank == 0) {
        printf("Parallel execution time: %f seconds\n", run_time);
    }

    MPI_Finalize();
    return 0;
}