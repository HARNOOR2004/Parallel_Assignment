# ASS2_Q4_mpi.c - Heat Distribution Simulation

# Description:

Simulates the spread of heat across a 1D grid over multiple iterations. Each process updates a portion of the grid and exchanges boundary values with its neighbors to model heat diffusion over time.

# Compilation And Execution:
``` sh 
mpicc ASS2_Q4_mpi.c -o heat_simulation
mpirun -np <num_processes> ./heat_simulation ```