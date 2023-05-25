import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import sys
#import cProfile
#cProfile.run('foo()')


import numpy as np
from time import time, sleep
sys.path.append('../../')
from h2py.data.triangle_surface import Data, integral_inverse_r,integral_inverse_r3

def sq(n):
        #v_count = n**2
        #p_count = (n-1)**2*2
        v = np.array([[],[],[]])
        p = np.array([],dtype = np.int32)
        for i_v in xrange(n):
            for j_v in xrange(n):
                v = np.concatenate((v,[[float(i_v)/n],[float(j_v)/n],[0]]),axis = 1)
        for i_p in xrange(n-1):
            for j_p in xrange(n-1):
            	p = np.concatenate((p,[3,i_p*n+j_p,(i_p+1)*n+j_p,i_p*n+j_p+1]),axis = 0)
                p = np.concatenate((p,[3,(i_p+1)*n+j_p+1,(i_p+1)*n+j_p,i_p*n+j_p+1]),axis = 0)
        #p = np.array(p,dtype = np.int)
        #import ipdb; ipdb.set_trace()
        problem = Data(v, p)
        problem.dim = 3
        return problem

problem = sq(10)
#problem = Data.from_dat('Geodat.dat')#sq(N)
#import ipdb; ipdb.set_trace()
func0 = integral_inverse_r3
#problem.count = 100
#problem.vertex = problem.vertex[:,:100]
from h2py.main import Problem
print 'building tree'
surf = Problem(func0, problem, block_size = 50,tol = 0)
#import ipdb; ipdb.set_trace()
print 'done'
print 'MCBH-SVD'
t0 = time()
surf.gen_queue(symmetric = False)
#import ipdb; ipdb.set_trace()
surf.factorize('h2', tau = 1e-2, iters = 1)
#import ipdb; ipdb.set_trace()
#print 'memory consumption: ' + str(surf.factor.nbytes()/1024./1024) + ' MB'
print 'Original solving', time() - t0
from  h2py.core.ts import hyp
from  h2py.core.ts import hyp_test
from  h2py.core.ts import show_hyp
from  h2py.core.ts import info_hyp
#surf.factor.svdcompress(1e-3, verbose=1)
#import ipdb; ipdb.set_trace()
hyp(surf)
test_vec = np.arange(problem.count)*0.01# np.ones(problem.count)#np.arange(problem.count)#np.zeros(problem.count)#
dot_test = surf.dot(test_vec, dasha_debug=1)
#import ipdb; ipdb.set_trace()
hyp_test(surf,dot_test,test_vec)
####import ipdb; ipdb.set_trace()





import scipy.io
from scipy.sparse.linalg import gmres
import scipy as sp


res = []
Pmat = scipy.io.mmread('m_rhs_P/rPSE13.mtx')
Pmat = sp.sparse.csc_matrix(Pmat)
Pmat_ILU = scipy.sparse.linalg.spilu(Pmat, drop_tol=1e-4, fill_factor= 10, drop_rule=None, permc_spec=None, diag_pivot_thresh=None, relax=None, panel_size=None, options=None)

#Pmat_LU = scipy.sparse.linalg.splu(Pmat,  permc_spec=None, diag_pivot_thresh=None, drop_tol=None, relax=None, panel_size=None, options={})
def simple_print(r):
    print np.linalg.norm(r)
    res.append(r)
def dot_fun(v):

	ans  = surf.dot(v, dasha_debug=1)
	#import ipdb; ipdb.set_trace()
	return ans[:v.shape[0]]
	
def prec(v):
	#t0 = time.time() 
	vec = np.concatenate([v,np.zeros(Pmat.shape[0]-v.shape[0])])
	ans = Pmat_ILU.solve(vec)
	#ans = sp.sparse.linalg.spsolve(Pmat,vec)
	#import ipdb; ipdb.set_trace()
	#print "iter time:", time.time() - t0
	return ans[:v.shape[0]]	
	
def GMRES_scipy():
	W = np.fromfile('m_rhs_P/W13')
	#Q = np.fromfile('Q')
	
	lo_dot = sp.sparse.linalg.LinearOperator((W.shape[0],W.shape[0]), dot_fun ,dtype = W.dtype)
	lo_prec = sp.sparse.linalg.LinearOperator((W.shape[0],W.shape[0]), prec ,dtype = W.dtype)
	#import ipdb; ipdb.set_trace()
	#ld = lo_dot.matvec(W)
	#lp = lo_prec.matvec(W)
        #import ipdb; ipdb.set_trace()
	#print "dot time:", time.time() - t0
	t0 = time() 
	ans = gmres(lo_dot, np.ones(W.shape[0]), np.zeros(W.shape[0]), tol = 1e-20, restart = 100000, maxiter = 2000,xtype=None, M = lo_prec, callback = simple_print)#M = lo_prec,
	
	print "GMRES time:", time() - t0
	
#import ipdb; ipdb.set_trace()	
GMRES_scipy()   


res = np.array(res)
res.tofile('resAN') 
#Direct_solver()    
#GMRES_SPARSKIT(drop_tol=1e-2)    
import ipdb; ipdb.set_trace()
