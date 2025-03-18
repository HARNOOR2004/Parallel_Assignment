#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

static long num_steps = 100000;
double step;

int main(int argc, char **argv) {
    int rank, size, i;
    double x, sum = 0.0, pi, partial_sum;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0) {
        step = 1.0 / (double)num_steps;
    }
    
    MPI_Bcast(&step, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    long chunk_size = num_steps / size;
    long start = rank * chunk_size;
    long end = start + chunk_size;
    
    partial_sum = 0.0;
    for (i = start; i < end; i++) {
        x = (i + 0.5) * step;
        partial_sum += 4.0 / (1.0 + x * x);
    }
    
    MPI_Reduce(&partial_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        pi = step * sum;
        printf("Approximate value of Pi: %f\n", pi);
    }
    
    MPI_Finalize();
    return 0;
}
