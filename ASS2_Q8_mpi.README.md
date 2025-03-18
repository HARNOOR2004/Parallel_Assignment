# ASS2_Q8_mpi.c - Parallel Matrix Transposition 

# Description:

Implements matrix transposition using MPI. The matrix is divided into sections assigned to different processes, which exchange rows and columns to achieve transposition efficiently.

# Compilation And Execution:
``` sh 
mpicc ASS2_Q8_mpi.c -o matrix_transpose
mpirun -np <num_processes> ./matrix_transpose ```