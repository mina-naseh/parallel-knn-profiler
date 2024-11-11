#!/bin/bash -l
#SBATCH --job-name=knn_benchmark            # Job name
#SBATCH --output=benchmark_output_hpc_copy2.txt   # Output file
#SBATCH --error=benchmark_error_hpc_copy2.txt     # Error file
#SBATCH --nodes=1                           # Number of nodes
#SBATCH --ntasks=4                          # Number of MPI tasks (processes)
#SBATCH --cpus-per-task=8                   # CPU cores per MPI task (for threading)
#SBATCH --time=05:00:00                     # Maximum runtime (hh:mm:ss)
#SBATCH --partition=batch                   # Partition/queue name

# Load modules and activate Python environment
module purge
module load lang/Python
module load mpi/OpenMPI
source .venv/bin/activate

# Set the number of benchmarking runs and threads per MPI task
RUNS=5
THREADS=8

# Run the benchmarking script with the specified arguments
mpiexec -n 4 python benchmarking.py --num_runs $RUNS --num_threads $THREADS
