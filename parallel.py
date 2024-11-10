from mpi4py import MPI
import numpy as np
from KNNClassifier import KNNClassifier

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Training data setup
rows = 100000
cols = 500
np.random.seed(699)
X_train = np.random.rand(rows, cols)
y_train = np.random.randint(2, size=rows)

# Initialize the classifier and fit data
k = 2
knn = KNNClassifier(k=k)
knn.fit(X_train, y_train)

# Generate test data indices (we'll use the same method but distribute across processes)
test_size = 1000
X_test = np.random.randint(rows, size=test_size)

# Split the test data across MPI processes
local_X_test = np.array_split(X_test, size)[rank]

# Each process computes predictions for its subset of test data
local_predictions = knn.predict(X_train[local_X_test])

# Gather results at the root process
all_predictions = comm.gather(local_predictions, root=0)

if rank == 0:
    # Flatten and combine predictions from all processes
    final_predictions = np.concatenate(all_predictions)
    
    # Calculate the accuracy
    correct_predictions = np.sum(y_train[X_test] == final_predictions)
    print(f'Correct Predictions: {correct_predictions}')
    print(f'Total Test Samples: {test_size}')
    print(f'Accuracy: {correct_predictions / test_size:.2%}')


# brew install open-mpi
# uv pip install mpi4py


