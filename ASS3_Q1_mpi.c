#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N (1 << 16)  // 2^16 elements
#define A 2.5        // Scalar multiplier

void daxpy_serial(double *x, double *y) {
    for (int i = 0; i < N; i++) {
        x[i] = A * x[i] + y[i];
    }
}

void daxpy_parallel(double *x, double *y, int rank, int size) {
    int local_n = N / size;
    double *local_x = (double *)malloc(local_n * sizeof(double));
    double *local_y = (double *)malloc(local_n * sizeof(double));
    
    MPI_Scatter(x, local_n, MPI_DOUBLE, local_x, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(y, local_n, MPI_DOUBLE, local_y, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    for (int i = 0; i < local_n; i++) {
        local_x[i] = A * local_x[i] + local_y[i];
    }
    
    MPI_Gather(local_x, local_n, MPI_DOUBLE, x, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    free(local_x);
    free(local_y);
}

int main(int argc, char **argv) {
    int rank, size;
    double *x, *y, start_time, serial_time, parallel_time;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0) {
        x = (double *)malloc(N * sizeof(double));
        y = (double *)malloc(N * sizeof(double));
        for (int i = 0; i < N; i++) {
            x[i] = rand() % 100;
            y[i] = rand() % 100;
        }
        
        start_time = MPI_Wtime();
        daxpy_serial(x, y);
        serial_time = MPI_Wtime() - start_time;
        printf("Serial execution time: %f seconds\n", serial_time);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        start_time = MPI_Wtime();
    }
    
    daxpy_parallel(x, y, rank, size);
    
    if (rank == 0) {
        parallel_time = MPI_Wtime() - start_time;
        printf("Parallel execution time: %f seconds\n", parallel_time);
        printf("Speedup: %f\n", serial_time / parallel_time);
        free(x);
        free(y);
    }
    
    MPI_Finalize();
    return 0;
}