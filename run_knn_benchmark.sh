#!/bin/bash -l
#SBATCH --job-name=knn_benchmark            # Job name
#SBATCH --output=benchmark_output_hpc.txt   # Output file
#SBATCH --error=benchmark_error_hpc.txt     # Error file
#SBATCH --nodes=1                           # Number of nodes
#SBATCH --ntasks=2                          # Number of MPI tasks (processes)
#SBATCH --cpus-per-task=4                   # CPU cores per MPI task (used for threading)
#SBATCH --time=05:00:00                     # Maximum runtime (hh:mm:ss)
#SBATCH --partition=batch                   # Partition/queue name (adjust based on your HPC system)

# Load the necessary MPI module
print_error_and_exit() { echo "***ERROR*** $*"; exit 1; }
module purge || print_error_and_exit "No 'module' command"
module load lang/Python
module load mpi/OpenMPI
source .venv/bin/activate

# Set the number of runs and threads
RUNS=30
THREADS=4

# Run the benchmarking script with command-line arguments
mpiexec -n 2 python benchmarking.py --num_runs $RUNS --num_threads $THREADS

# # Run the benchmarking script with mpiexec
# mpiexec -n 2 python benchmarking.py
