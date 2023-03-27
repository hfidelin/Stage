# Mise en place d'un solveur pour matrice $\mathcal{H} ^ 2$

Codes Python réalisés lors de mon stage de master sur la mise enplace d'un solveur pour matrice $\mathcal{H}^2$.

## Installation de $h2tools$

Pour ces codes, il est nécessaire d'installer différents package Python :

* pip 
* numpy
* numba
* scipy
* cython
* maxvolpy
* pypropack
* h2tools

Un tutorial est (en cours) disponible dans le rapport de ce stage.



## Analyse du produit matrice-vecteur pour une matrice $\mathcal{H} ^ 2$

La classe 'H2Matrix' contient déjà un produit scalaire matrice-vecteur nommé \verb+.dot()+. Il est alors intéressant d'étudier l'erreur commise par cette opération pour une matrice aléatoire de taille $N \times N$, et en fonction de $\tau > 0$


![erreur_mat_vec](./Images/Err_Prod_Mat_Vec_log_N.png)

On lit alors qu'en choisissant $\tau < 10{-5}$ on obtient l'erreur la plus faible possible. Dans le reste des calculs, on prendra alors $\mathbf{\tau = 10^{-5}}$ 

## Solveur itératif de Krylov

La classe 'H2Matrix' étant un format data-sparse, elle ne donne accès qu'à très peu d'information quant à la matrice. Comme on l'a vu plus haut, cette classe embarque un produit matrice-vecteur, néanmoins aucune autre opérations n'est disponible. Il est donc sensé d'utiliser des algorithmes itératifs dans l'objectif de résoudre :

$$ Ax = b $$

Avec :
* $A$ matrice $\mathcal{H} ^ 2 (\mathbb{R} ^ {N \times N})$
* $b\in\mathbb{R} ^ {N }$ vecteur source 
* $x\in\mathbb{R} ^ {N }$ vecteur solution

Nous allons nous intéresser aux solveurs itératifs de Krylov, en particulier les algorithmes GMRES car ils ne nécessitent que d'un produit matrice-vecteur pour fonctionner.

Dans un premier temps, utilisons la fonction *scipy.sparse.linalg.gmres* ( ou une de ses variantes *scipy.sparse.linalg.lgmres*) afin de réaliser un tel algorithme.

Il est à noter que pour utiliser ces fonctions il faut rajouter un un attribut dans le fichier $h2matrix.py$. Il faudra ajouter :

```
def mat_vec(self,x):
	return self.dot(x)
```

Afin que *scipy* puisse utiliser un produit matrice-vecteur.
