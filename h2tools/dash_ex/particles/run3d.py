import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import sys
import numpy as np
from time import time
sys.path.append('../../')
from h2py.data.particles import Data, inv_dist_int,inv_distance#



problem = Data.from_txt_file('data3d.txt')
func0 = inv_dist_int #inv_distance#
#import ipdb; ipdb.set_trace()
problem.count = 10
problem.vertex = problem.vertex[:,:10]
N = 100
np.random.seed(100)
problem = Data(np.random.rand(3, N))
func0 = inv_distance
from h2py.main import Problem
print 'building tree'
surf = Problem(func0, problem, block_size = 3)
surf.gen_queue(symmetric = 0)
print 'done'
print 'MCBH-SVD'
surf.factorize('h2', tau = 1e-1, iters = 1)
print 'memory consumption: ' + str(surf.factor.nbytes()/1024./1024) + ' MB'

from  h2py.core.ts import hyp
from  h2py.core.ts import sparse_gipermatrix_test
from  h2py.core.ts import show_hyper
from  h2py.core.ts import info_hyper
from  h2py.core.ts import solving
hyp(surf)
test_vec = np.ones(problem.count)
dot_test = surf.factor.dot(test_vec, dasha_debug=1)
sparse_gipermatrix_test(surf,dot_test,test_vec)
#show_hyper(surf)
solving(surf)
#info_hyper(surf)

