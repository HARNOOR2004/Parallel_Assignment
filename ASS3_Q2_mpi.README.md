# ASS3_Q2_mpi.c - Parallel Approximation of Pi

# Description:

Uses numerical integration to estimate the value of Pi. The computation is divided among multiple processes, each calculating a portion of the integral. The results are then combined using MPI_Reduce to produce the final approximation.

# Compilation And Execution:
``` sh 
mpicc ASS3_Q2_mpi.c -o pi_approx  
mpirun -np <num_processes> ./pi_approx  
 ```