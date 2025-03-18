# ASS2_Q2_mpi.c - Parallel Matrix Multiplication

# Description:

Implements matrix multiplication in parallel using MPI. The master process distributes rows of the first matrix among worker processes, which compute their portion of the result before gathering the final output. This approach optimizes large matrix computations.

# Compilation And Execution:
``` sh 
mpicc ASS2_Q2_mpi.c -o matrix_multiply -fopenmp
mpirun -np <num_processes> ./matrix_multiply ```