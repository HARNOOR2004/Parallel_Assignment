# ASS2_Q6_mpi.c - Parallel Dot Product Calculation 

# Description:

Performs the dot product of two vectors in parallel using MPI. The vectors are divided among multiple processes, each computing a partial sum, which is then aggregated using MPI_Reduce to get the final result.

# Compilation And Execution:
``` sh 
mpicc ASS2_Q6_mpi.c -o dot_product
mpirun -np <num_processes> ./dot_product ```