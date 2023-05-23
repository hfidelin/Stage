import numpy as np
from random import randint
import MMMF
import time
import mesh as m
import scipy.sparse.linalg as spla
import tt
from tt.amen import amen_solve
from numpy import linalg as LA
from inverse import inverse as inv
from inverse import inverse_LYA as invLYA
from scipy.sparse.linalg import gmres


###########################################################################
# script trouve pour affiche les iterations du gmres python
class gmres_counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1
        #if self._disp:
            #print('iter %3i\trk = %s' % (self.niter, str(rk)))


######################################################################




#choix_methode = 1
#Mom = MMM.solve(choix_methode,b,A,n)  
#print("Resolution systeme lineaire (",tr," s)")
print(" ")
print(" -- Resolution Full  --- ")
#Momp,info=spla.gmres(A,b)
tr = time.time()
xref=LA.solve(A,b) # Solution FULL
tr = time.time() - tr
print("Temps Resolution Full ",tr," s")
#print(info)

print(" ")
print(" -- Resolution GMRES Full  --- ")
tr = time.time()
x,info=spla.gmres(A,b)
tr = time.time() - tr
print("Temps Resolution GMRES ",tr," s")



print(" ")
print(" -- Amen-solve  --- ")
tr = time.time()
xtt= amen_solve(Att,btt,btt,eps,verb=1) # Linear system solution using AMEN
tr = time.time() - tr
print("Temps solveur amen ",tr," s")
#xt=xtt.full() 
#x=np.reshape(xt,N,order='F')
print("erreur Amen=",LA.norm(x-xref)/LA.norm(xref))
print(" ")



#M_x = lambda x: np.matmul(A, x)
AttOP_x = lambda x: Att*x
AttOP = spla.LinearOperator((N, N), AttOP_x)


print(" -- Resolution GMRES QTT  --- ")
tr = time.time()
counter = gmres_counter()
x,info=gmres(AttOP,b,tol=eps,restart=int(N/2),maxiter=N,callback=counter)
tr = time.time() - tr
print("Temps Resolution GMRES QTT ",tr," s")
print("Iterations ={}".format(counter.niter))
print("erreur GMRES(QTT) =",LA.norm(x-xref)/LA.norm(xref))
print(" ")




print(" -- Construction PREC  --- ")
epsprec=1.e-2
tr = time.time()
Ptt=Att.round(epsprec)
Pm1=invLYA(Ptt,epsprec)
tr = time.time() - tr
print("Temps construction PREC  (",tr," s)")



Pm1OP_x = lambda x: Pm1*x
Pm1OP = spla.LinearOperator((N, N), Pm1OP_x)

print(" ")
print(" -- Resolution GMRES Preconditionne  --- ")
print(" ")
tr = time.time()
counter = gmres_counter()
x,info=gmres(AttOP,b,tol=1e-6,restart=int(N/2),maxiter=N,M=Pm1OP,callback=counter)
#x=Pm1*b
tr = time.time() - tr
print("Temps Resolution GMRES Preconditionne (",tr," s)")
print("Iterations ={}".format(counter.niter))
print(" ")
print("erreur GMRES (Prec) =",LA.norm(x-xref)/LA.norm(xref))


# Remplacer par validation de la solution
#Am1=inv(Att,eps)




 
