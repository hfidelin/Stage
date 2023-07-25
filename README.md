---
bibliography: reference.bib
---

# Mise en place d'un solveur pour matrice $\mathcal{H} ^ 2$

Codes Python réalisés lors de mon stage de master sur la mise enplace d'un solveur pour matrice $\mathcal{H}^2$.

## Installation de $h2tools$

Afin d'utiliser, les fichiers de ce dépôt GitHub, voilà un rapide tutoriel pour l'installation des différents packages nécessaires.

Nous recommendons vivement l'utilisation de $pip$ (https://pip.pypa.io/en/stable/installation/) pour l'installation des packages suivant :
 
* numpy
* numba
* scipy
* cython
* maxvolpy

Par exemple à l'aide de la commande :

```
pip install cython
```


Une fois ces packages installés, on peut alors installer $h2tools$ :

1. Télécharger ce dépôt GitHub via :

```
git clone https://github.com/hfidelin/Stage.git
```

2. Se placer dans le dépot GitHub 'Stage' qui vient d'être téléchargé

3. Installer $h2tools$ via pip en rentrant :

```
pip install h2tools/
```

(**ATTENTION** : ne pas oublier le '/' à la fin de 'h2tools/', sinon vous installerez un package différent de celui du Git)
