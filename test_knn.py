from KNNClassifier import KNNClassifier
import numpy as np

# Example with random data
rows = 100000
cols = 500
np.random.seed(699)
X_train = np.random.rand(rows*cols).reshape((rows,cols))
y_train = np.random.randint(2, size=rows)
print(f'X_train shape {X_train.shape} - y_train shape {y_train.shape}')

knn = KNNClassifier(k=2)
knn.fit(X_train, y_train)

# Create random indices to test
test_size = 1000
X_test = np.random.randint(rows, size=test_size)

# Generate Predictions
predictions = knn.predict(X_train[X_test])
# print(f'Prediction {predictions}')
# print(f'Label      {y_train[X_test]}')

# Calculate the number of equal elements
print(f'correct {np.sum(y_train[X_test] == predictions)}')


# install uv An extremely fast Python package manager.
# uv venv
# uv pip install line_profiler
# kernprof -l -v test_knn.py
# python -m line_profiler test_knn.py.lprof > profile_report.txt

# brew install open-mpi
# uv pip install mpi4py
# sysctl -n hw.ncpu

# module load mpi/OpenMPI
