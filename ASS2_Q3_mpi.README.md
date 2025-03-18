# ASS2_Q3_mpi.c - Parallel Odd-Even Sorting

# Description:

Uses MPI to implement the Odd-Even Sort algorithm for sorting an array in parallel. The data is distributed among multiple processes, which exchange and compare elements iteratively until the entire array is sorted.

# Compilation And Execution:
``` sh 
mpicc ASS2_Q3_mpi.c -o odd_even_sort
mpirun -np <num_processes> ./odd_even_sort ```