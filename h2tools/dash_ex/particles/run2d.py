import sys
import numpy as np
from time import time





sys.path.append('../../')

from h2py.data.particles import Data,  inv_distance #log_distance
t = 1500
problem = Data.from_txt_file('data2d.txt')

func0 = inv_distance #log_distance
problem.count = t
problem.vertex = problem.vertex[:,:t]
from h2py.main import Problem
print 'building tree'
surf = Problem(func0, problem, block_size = 20)
print 'done'
print 'MCBH-SVD'
surf.factorize('h2', tau = 1e-2, iters = 1)
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
solving(surf)
show_hyper(surf)
#info_hyper(surf)
import pdb; pdb.set_trace()
