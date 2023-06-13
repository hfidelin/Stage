import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import LinearOperator

# Define a CSC matrix A
A = csc_matrix([[1, 0], [0, 1]])

# Creating Linear operator from another matrix
B = np.array([[10, 0], [0, 10]] )
lin_op_B = LinearOperator(B.shape, matvec=B.dot)

# Perform the matrix-matrix multiplication
result = A @ lin_op_B

print(result)