# ASS2_Q7_mpi.c - Parallel Prefix Sum (Scan)Calculation 

# Description:

Computes the prefix sum (cumulative sum) of an array using MPI_Exscan. Each process calculates its local prefix sum, and MPI ensures proper accumulation to maintain the order of computation across distributed processes.

# Compilation And Execution:
``` sh 
mpicc ASS2_Q7_mpi.c -o prefix_sum
mpirun -np <num_processes> ./prefix_sum ```