from mpi4py import MPI
import argparse
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from KNNClassifier import KNNClassifier  # Import the unmodified class
from time import time, process_time


# Define and parse command-line arguments
parser = argparse.ArgumentParser(description="KNN Benchmarking Script")
parser.add_argument("--num_runs", type=int, default=5, help="Number of benchmarking runs")
parser.add_argument("--num_threads", type=int, default=4, help="Number of threads per MPI task")
args = parser.parse_args()

num_runs = args.num_runs
num_threads = args.num_threads


# Helper function for multi-threaded distance calculations
def predict_single_test_point(knn, x, num_threads):
    """
    Predict the label for a single test point using multi-threaded distance calculations.

    Arguments:
    knn -- the KNNClassifier instance
    x -- the single test point for which we are predicting the label
    num_threads -- the number of threads to use for parallelizing distance calculations

    Returns:
    The predicted label for the test point.
    """
    # Split the training data into subsets for each thread
    training_subsets = np.array_split(knn.X_train, num_threads)
    # Use ThreadPoolExecutor to parallelize distance calculations within each subset
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        distances_parts = executor.map(
            lambda subset: [knn.euclidean_distance(x, x_train) for x_train in subset],
            training_subsets
        )
    # Combine distances from each thread and select the k-nearest neighbors
    distances = np.concatenate(list(distances_parts))
    k_indices = np.argsort(distances)[:knn.k]
    k_nearest_labels = [knn.y_train[i] for i in k_indices]
    return np.bincount(k_nearest_labels).argmax()

# Main function for parallel predictions
def parallel_predict(knn, X_test, num_threads):
    """
    Perform parallel predictions on the test dataset using MPI and threading.

    Arguments:
    knn -- the KNNClassifier instance
    X_test -- the test dataset
    num_threads -- the number of threads to use for intra-process parallelization

    Returns:
    All predictions gathered from each process on the root process, or None for non-root processes.
    """
    # Initialize MPI and get process information
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Distribute the test dataset across MPI processes
    local_X_test = np.array_split(X_test, size)[rank]

    # Each process predicts for its subset of test points using multi-threading
    local_predictions = [predict_single_test_point(knn, x, num_threads) for x in local_X_test]

    # Gather predictions from all processes to the root process
    all_predictions = comm.gather(local_predictions, root=0)

    # Only the root process will receive the full set of predictions
    if rank == 0:
        return np.concatenate(all_predictions)
    else:
        return None

# Benchmarking
if __name__ == "__main__":
    # Initialize data
    rows, cols = 100000, 500
    np.random.seed(699)
    X_train = np.random.rand(rows, cols)
    y_train = np.random.randint(2, size=rows)
    test_size = 1000
    X_test_indices = np.random.randint(rows, size=test_size)
    X_test = X_train[X_test_indices]

    # Instantiate and fit KNNClassifier
    knn = KNNClassifier(k=2)
    knn.fit(X_train, y_train)

    # Open report file to save benchmarking results
    report_file = open("benchmark_report.txt", "w")

    # Set the number of benchmarking runs
    num_runs = num_runs
    sequential_real_times = []
    sequential_cpu_times = []
    parallel_real_times = []
    parallel_cpu_times = []

    # Sequential Benchmarking (run only on root process)
    if MPI.COMM_WORLD.Get_rank() == 0:
        for _ in range(num_runs):
            start_real = time()  # Start real time (wall-clock)
            start_cpu = process_time()  # Start CPU time

            # Perform prediction using the sequential approach
            sequential_predictions = knn.predict(X_test)

            end_real = time()  # End real time
            end_cpu = process_time()  # End CPU time

            # Record real time and CPU time for this run
            sequential_real_times.append(end_real - start_real)
            sequential_cpu_times.append(end_cpu - start_cpu)

            # Verify correctness
            correct_sequential = np.sum(y_train[X_test_indices] == sequential_predictions)
        
        # Output the correctness for the sequential version
        seq_report = f'Sequential correct: {correct_sequential}\n'
        print(seq_report)
        report_file.write(seq_report)

    # Parallel Benchmarking
    for _ in range(num_runs):
        start_real = time()  # Start real time
        start_cpu = process_time()  # Start CPU time

        # Perform prediction using the parallel approach
        parallel_predictions = parallel_predict(knn, X_test, num_threads=num_threads)

        end_real = time()  # End real time
        end_cpu = process_time()  # End CPU time

        # Record real time and CPU time for this run, only on root process
        if MPI.COMM_WORLD.Get_rank() == 0:
            parallel_real_times.append(end_real - start_real)
            parallel_cpu_times.append(end_cpu - start_cpu)

            # Check correctness
            assert np.array_equal(sequential_predictions, parallel_predictions), "Mismatch between parallel and sequential predictions!"
            correct_parallel = np.sum(y_train[X_test_indices] == parallel_predictions)
    
    # Calculate and display results (only root process)
    if MPI.COMM_WORLD.Get_rank() == 0:
        # Calculate averages and standard deviations for sequential times
        seq_avg_real_time = np.mean(sequential_real_times)
        seq_std_real_time = np.std(sequential_real_times)
        seq_avg_cpu_time = np.mean(sequential_cpu_times)
        seq_std_cpu_time = np.std(sequential_cpu_times)
        seq_time_report = (
            f"Sequential real times: {sequential_real_times}\n"
            f"Sequential CPU times: {sequential_cpu_times}\n"
            f"Sequential time (avg real): {seq_avg_real_time:.4f} s, std: {seq_std_real_time:.4f} s\n"
            f"Sequential time (avg CPU): {seq_avg_cpu_time:.4f} s, std: {seq_std_cpu_time:.4f} s\n"
        )
        print(seq_time_report)
        report_file.write(seq_time_report)
        
        # Calculate averages and standard deviations for parallel times
        par_avg_real_time = np.mean(parallel_real_times)
        par_std_real_time = np.std(parallel_real_times)
        par_avg_cpu_time = np.mean(parallel_cpu_times)
        par_std_cpu_time = np.std(parallel_cpu_times)
        par_time_report = (
            f"Parallel real times: {parallel_real_times}\n"
            f"Parallel CPU times: {parallel_cpu_times}\n"
            f"Parallel time (avg real): {par_avg_real_time:.4f} s, std: {par_std_real_time:.4f} s\n"
            f"Parallel time (avg CPU): {par_avg_cpu_time:.4f} s, std: {par_std_cpu_time:.4f} s\n"
        )
        print(par_time_report)
        report_file.write(par_time_report)
        
        # Correctness verification for parallel results
        correct_report = f'Parallel correct: {correct_parallel}\n'
        print(correct_report)
        report_file.write(correct_report)

        # Calculate speed-up based on real time
        speed_up = seq_avg_real_time / par_avg_real_time
        speed_up_report = f'Speed-up (real time): {speed_up:.2f}x\n'
        print(speed_up_report)
        report_file.write(speed_up_report)

    # Close the report file to save benchmarking results
    if MPI.COMM_WORLD.Get_rank() == 0:
        report_file.close()