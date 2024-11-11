from mpi4py import MPI
import argparse
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor
from KNNClassifier import KNNClassifier
from time import time, process_time

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(),  # Log to console
                        logging.FileHandler("benchmark_log.txt")  # Log to file
                    ])

# Define and parse command-line arguments
parser = argparse.ArgumentParser(description="KNN Benchmarking Script")
parser.add_argument("--num_runs", type=int, default=5, help="Number of benchmarking runs")
parser.add_argument("--num_threads", type=int, default=4, help="Number of threads per MPI task")
args = parser.parse_args()

num_runs = args.num_runs
num_threads = args.num_threads

# Helper function for multi-threaded distance calculations
def predict_single_test_point(knn, x, num_threads):
    logging.info(f"Predicting for single test point with {num_threads} threads.")
    training_subsets = np.array_split(knn.X_train, num_threads)
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        distances_parts = executor.map(
            lambda subset: [knn.euclidean_distance(x, x_train) for x_train in subset],
            training_subsets
        )
    distances = np.concatenate(list(distances_parts))
    k_indices = np.argsort(distances)[:knn.k]
    k_nearest_labels = [knn.y_train[i] for i in k_indices]
    return np.bincount(k_nearest_labels).argmax()

# Main function for parallel predictions
def parallel_predict(knn, X_test, num_threads):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    logging.info(f"Process {rank}/{size} is predicting with {num_threads} threads.")
    local_X_test = np.array_split(X_test, size)[rank]
    local_predictions = [predict_single_test_point(knn, x, num_threads) for x in local_X_test]
    all_predictions = comm.gather(local_predictions, root=0)

    if rank == 0:
        logging.info("All predictions gathered on the root process.")
        return np.concatenate(all_predictions)
    else:
        return None

# Benchmarking
if __name__ == "__main__":
    rows, cols = 100000, 500
    np.random.seed(699)
    X_train = np.random.rand(rows, cols)
    y_train = np.random.randint(2, size=rows)
    test_size = 1000
    X_test_indices = np.random.randint(rows, size=test_size)
    X_test = X_train[X_test_indices]

    knn = KNNClassifier(k=2)
    knn.fit(X_train, y_train)
    report_file = open("benchmark_report.txt", "w")

    sequential_real_times = []
    sequential_cpu_times = []
    parallel_real_times = []
    parallel_cpu_times = []

    # Sequential Benchmarking (run only on root process)
    if MPI.COMM_WORLD.Get_rank() == 0:
        logging.info("Starting sequential benchmarking.")
        for i in range(num_runs):
            logging.info(f"Sequential run {i+1}/{num_runs}.")
            start_real = time()
            start_cpu = process_time()

            sequential_predictions = knn.predict(X_test)

            end_real = time()
            end_cpu = process_time()

            sequential_real_times.append(end_real - start_real)
            sequential_cpu_times.append(end_cpu - start_cpu)

            correct_sequential = np.sum(y_train[X_test_indices] == sequential_predictions)
        
        seq_report = f'Sequential correct: {correct_sequential}\n'
        logging.info(seq_report)
        report_file.write(seq_report)

    # Parallel Benchmarking
    logging.info("Starting parallel benchmarking.")
    for i in range(num_runs):
        logging.info(f"Parallel run {i+1}/{num_runs}.")
        start_real = time()
        start_cpu = process_time()

        parallel_predictions = parallel_predict(knn, X_test, num_threads=num_threads)

        end_real = time()
        end_cpu = process_time()

        if MPI.COMM_WORLD.Get_rank() == 0:
            parallel_real_times.append(end_real - start_real)
            parallel_cpu_times.append(end_cpu - start_cpu)
            correct_parallel = np.sum(y_train[X_test_indices] == parallel_predictions)
    
    # Display and save results (only root process)
    if MPI.COMM_WORLD.Get_rank() == 0:
        logging.info("Calculating and logging benchmarking results.")
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
        logging.info(seq_time_report)
        report_file.write(seq_time_report)
        
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
        logging.info(par_time_report)
        report_file.write(par_time_report)
        
        correct_report = f'Parallel correct: {correct_parallel}\n'
        logging.info(correct_report)
        report_file.write(correct_report)

        speed_up = seq_avg_real_time / par_avg_real_time
        speed_up_report = f'Speed-up (real time): {speed_up:.2f}x\n'
        logging.info(speed_up_report)
        report_file.write(speed_up_report)

    if MPI.COMM_WORLD.Get_rank() == 0:
        report_file.close()
