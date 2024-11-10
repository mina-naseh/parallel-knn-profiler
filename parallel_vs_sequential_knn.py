from mpi4py import MPI
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from KNNClassifier import KNNClassifier  # Import the unmodified class

from timeit import default_timer as timer

# Helper function for multi-threaded distance calculations
def predict_single_test_point(knn, x, num_threads):
    # Split training data for multi-threaded distance calculation
    training_subsets = np.array_split(knn.X_train, num_threads)

    # Calculate distances in parallel using threads
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        distances_parts = executor.map(
            lambda subset: [knn.euclidean_distance(x, x_train) for x_train in subset],
            training_subsets
        )

    # Combine distances across threads
    distances = np.concatenate(list(distances_parts))
    k_indices = np.argsort(distances)[:knn.k]
    k_nearest_labels = [knn.y_train[i] for i in k_indices]
    return np.bincount(k_nearest_labels).argmax()

# Main function for parallel predictions
def parallel_predict(knn, X_test, num_threads):
    # Set up MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Split the test data across MPI processes
    local_X_test = np.array_split(X_test, size)[rank]

    # Each process predicts for its subset of test points
    local_predictions = [predict_single_test_point(knn, x, num_threads) for x in local_X_test]

    # Gather predictions from all processes at the root process
    all_predictions = comm.gather(local_predictions, root=0)

    if rank == 0:
        # Flatten and return combined predictions
        return np.concatenate(all_predictions)
    else:
        return None

if __name__ == "__main__":
    # Initialize data
    rows, cols = 10000, 500
    np.random.seed(699)
    X_train = np.random.rand(rows, cols)
    y_train = np.random.randint(2, size=rows)
    test_size = 100
    X_test_indices = np.random.randint(rows, size=test_size)
    X_test = X_train[X_test_indices]

    # Instantiate and fit KNNClassifier
    knn = KNNClassifier(k=2)
    knn.fit(X_train, y_train)

    # Run sequential prediction only in the root process
    if MPI.COMM_WORLD.Get_rank() == 0:
        start = timer()
        sequential_predictions = knn.predict(X_test)
        end = timer()
        print(f'Sequential time: {end - start:.4f} seconds')
        correct_sequential = np.sum(y_train[X_test_indices] == sequential_predictions)
        print(f'Sequential correct: {correct_sequential}')

    # Run parallel prediction
    start = timer()
    parallel_predictions = parallel_predict(knn, X_test, num_threads=2)

    # Only the root process will print the comparison
    if MPI.COMM_WORLD.Get_rank() == 0:
        # Check if the parallel predictions match the sequential predictions
        assert np.array_equal(sequential_predictions, parallel_predictions), "Mismatch between parallel and sequential predictions!"
        end = timer()
        print(f'Parallel time: {end - start:.4f} seconds')

        # Calculate the number of correct predictions in parallel
        correct_parallel = np.sum(y_train[X_test_indices] == parallel_predictions)
        print(f'Parallel correct: {correct_parallel}')




