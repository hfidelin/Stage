"""
Fonctions permettant de vérifier (de préférence pour des problèmes linéaire
de petite taille) la validité d'un solveur en passant par un solveur itératif
de Krylov
"""
import numpy as np
import time
import matplotlib.pyplot as plt
from function_GMRES import init_particules_problem, solve_gmres
from h2tools.mcbh import mcbh
from scipy.sparse.linalg import LinearOperator
from h2tools.collections import particles


class gmres_counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            print('iter %3i\trk = %s' % (self.niter, str(rk)))



if __name__ == "__main__":

    np.random.seed(0)
    # Initialisation des paramètres du problème
    start = time.time()
    # N_vec = [100, 150, 200, 250, 300, 350, 400]
    #N_vec = [500 * i for i in range(1, 9)]
    #N_vec = [50 * i for i in range(1, 20)]
    #N_vec = [1000, 2000, 3000, 4000, 5000]
    #N_vec = [10, 25, 50, 75, 100]
    N_vec = [500 * i for i in range(1, 14, 2)]
    N_vec =[500, 1000, 1500, 2000, 2500, 3000, 4000, 5000] 
    for ndim in [1, 2, 3]: 
        print(f'DIMENSION = {ndim} ')
        X = []
        Y = []
        for N in N_vec:

            X.append(N)
            print(f"\n{ndim}D : N = {N}\n")
            position = np.random.randn(ndim, N)
            tau = 1e-9
            
            func = particles.inv_distance
        
            problem, L, A = init_particules_problem(position, func, block_size=100, full_matrix=True)   
            A += 100_000 * np.eye(N)
            print(f'\n{ndim}D :CONDITIONNEMENT : ', np.linalg.cond(A))
            A_h2 = mcbh(problem, tau=tau, iters=1, verbose=0)  #Matrice H2
            
            mv = A_h2.dot
            
            A_h2 = LinearOperator((N, N), matvec=mv)
            b = np.ones(N)

            count = gmres_counter()
            x_ref = np.linalg.solve(A, b)
            x, err = solve_gmres(N, A, b, x_ref, eps=tau, counter = count)
                
            Y.append(err)


        print(f"Temps d'exécution : {time.time() - start}")

        
        plt.loglog(X, Y, label=f'{ndim}D', linewidth=2)

    plt.title("Erreur solveur GMRES")
    plt.xlabel("Nombre d'inconnu $N$")
    plt.ylabel(r"Valeur de $\|x - \tilde{x} \|_2$")    
    plt.ylim((1e-15, 1e4))
    plt.grid()
    plt.legend()
    plt.show()

   

    
    

        
