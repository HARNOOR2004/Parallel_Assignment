# ASS2_Q1_mpi.c - Monte Carlo Estimation of Pi

# Description:

This program approximates the value of Pi using the Monte Carlo method. Each process generates a set of random points and counts how many fall within the unit circle. The results are aggregated using MPI_Reduce to calculate the final estimate.

# Compilation And Execution:
``` sh 
mpicc ASS2_Q1_mpi.c -o montecarlo_pi -lm
mpirun -np <num_processes> ./montecarlo_pi ```