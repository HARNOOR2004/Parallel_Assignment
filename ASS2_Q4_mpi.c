#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 10       // Grid size
#define ITER 100   // Number of iterations
#define LEFT 0     // Left boundary value
#define RIGHT 100  // Right boundary value

void initialize_grid(double grid[N]) {
    grid[0] = LEFT;
    grid[N - 1] = RIGHT;
    for (int i = 1; i < N - 1; i++) {
        grid[i] = 0.0;
    }
}

void print_grid(double grid[N]) {
    for (int i = 0; i < N; i++) {
        printf("%6.2f ", grid[i]);
    }
    printf("\n");
}

int main(int argc, char **argv) {
    int rank, size;
    double grid[N], new_grid[N];
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0) {
        initialize_grid(grid);
        printf("Initial grid:\n");
        print_grid(grid);
    }
    
    MPI_Bcast(grid, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    for (int iter = 0; iter < ITER; iter++) {
        for (int i = rank + 1; i < N - 1; i += size) {
            new_grid[i] = 0.5 * (grid[i - 1] + grid[i + 1]);
        }
        
        MPI_Allgather(new_grid, N, MPI_DOUBLE, grid, N, MPI_DOUBLE, MPI_COMM_WORLD);
    }
    
    if (rank == 0) {
        printf("Final grid after %d iterations:\n", ITER);
        print_grid(grid);
    }
    
    MPI_Finalize();
    return 0;
}