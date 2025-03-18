# ASS2_Q5_mpi.c - Parallel Array Summation

# Description:

Computes the sum of an array using MPI. Each process calculates a local sum for its assigned portion of the array, and MPI_Reduce combines the results to obtain the global sum efficiently.

# Compilation And Execution:
``` sh 
mpicc ASS2_Q5_mpi.c -o sum_array
mpirun -np <num_processes> ./sum_array ```