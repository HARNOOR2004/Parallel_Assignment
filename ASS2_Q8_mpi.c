#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 4  // Matrix size (N x N)

void print_matrix(int matrix[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
}

int main(int argc, char **argv) {
    int rank, size;
    int matrix[N][N], transposed[N][N];
    int local_row[N];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        srand(42);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                matrix[i][j] = rand() % 10;
            }
        }
        printf("Original Matrix:\n");
        print_matrix(matrix);
    }

    MPI_Scatter(matrix, N, MPI_INT, local_row, N, MPI_INT, 0, MPI_COMM_WORLD);
    
    int local_col[N];
    for (int i = 0; i < N; i++) {
        local_col[i] = local_row[i];
    }
    
    MPI_Gather(local_col, N, MPI_INT, transposed, N, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Transposed Matrix:\n");
        print_matrix(transposed);
    }

    MPI_Finalize();
    return 0;
}