#!/bin/bash
#SBATCH --job-name=knn_benchmark            # Job name
#SBATCH --output=benchmark_output_hpc.txt   # Output file
#SBATCH --error=benchmark_error_hpc.txt     # Error file
#SBATCH --nodes=1                           # Number of nodes
#SBATCH --ntasks=2                          # Number of MPI tasks (processes)
#SBATCH --cpus-per-task=4                   # CPU cores per MPI task (used for threading)
#SBATCH --time=03:00:00                     # Maximum runtime (hh:mm:ss)
#SBATCH --partition=batch                   # Partition/queue name (adjust based on your HPC system)

# Load the necessary MPI module
module load mpi/OpenMPI

# Run the benchmarking script with mpiexec
mpiexec -n 2 python benchmarking.py
