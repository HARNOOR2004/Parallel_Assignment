#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

int is_prime(int num) {
    if (num < 2) return 0;
    for (int i = 2; i <= sqrt(num); i++) {
        if (num % i == 0) return 0;
    }
    return 1;
}

int main(int argc, char **argv) {
    int rank, size, max_value = 100;
    int num, result;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0) {
        int count = 2;
        int active_workers = size - 1;
        while (active_workers > 0) {
            MPI_Recv(&result, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (result > 0) {
                printf("Prime: %d\n", result);
            }
            if (count <= max_value) {
                MPI_Send(&count, 1, MPI_INT, result < 0 ? -result : rank, 0, MPI_COMM_WORLD);
                count++;
            } else {
                int stop_signal = -1;
                MPI_Send(&stop_signal, 1, MPI_INT, result < 0 ? -result : rank, 0, MPI_COMM_WORLD);
                active_workers--;
            }
        }
    } else {
        while (1) {
            num = 0;
            MPI_Send(&num, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Recv(&num, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (num == -1) break;
            result = is_prime(num) ? num : -rank;
            MPI_Send(&result, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }
    MPI_Finalize();
    return 0;
}