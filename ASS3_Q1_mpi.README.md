# ASS3_Q1_mpi.c - Parallel DAXPY Computation

# Description:

Performs the DAXPY operation (y = A * x + y) in parallel using MPI. The computation is distributed across processes using MPI_Scatter and MPI_Gather, allowing efficient processing of large datasets. The program also measures execution speedup compared to the serial version.

# Compilation And Execution:
``` sh 
mpicc ASS3_Q1_mpi.c -o daxpy
mpirun -np <num_processes> ./daxpy ```