# Mise en place d'un solveur pour matrice $\mathcal{H} ^ 2$

Codes Python réalisés lors de mon stage de master sur la mise enplace d'un solveur pour matrice H2.

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

[A COMPLETER]

Il est à noter que pour utiliser *scipy.sparse.linalg.gmres* il faut ajouter un un attribut dans le fichier $h2matrix.py$. Il faudra ajouter :

```
def mat_vec(self,x):
	return self.dot(x)
```

Afin que * scipy * puisse utiliser un produit matrice-vecteur.
