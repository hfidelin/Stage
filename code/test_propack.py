from pypropack import svdp
import numpy as np   
import time

start = time.time()
A = np.random.random((10, 20))
u, s, vt = svdp(A, k=3)  # compute top k singular values
np.set_printoptions(precision=3, suppress=True)
print(abs(np.dot(vt, vt.T)))
print(f"Temps de calcul : {time.time() - start}")