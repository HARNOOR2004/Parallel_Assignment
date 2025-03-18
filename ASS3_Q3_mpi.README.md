# ASS3_Q3_mpi.c - Distributed Prime Number Search

# Description:

A parallel algorithm to find prime numbers up to a specified limit. The master process distributes numbers among worker processes, which check for primality. Each worker returns results to the master, which collects and prints the primes.

# Compilation And Execution:
``` sh 
mpicc ASS3_Q3_mpi.c -o prime_finder -lm  
mpirun -np <num_processes> ./prime_finder   
 ```