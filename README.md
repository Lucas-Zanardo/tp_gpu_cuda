<div text-align="center">

# Compte Rendu TP

**Sujet:** "The power method on GPU", TP3 du Master 2 de Data Science de Lille ([pdf](./TP3_PowerGPU-etu.pdf))

Zanardo Lucas.

</div>

## Code

Pour compiler : 
```shell
$ make
```

Les executables sont `checker`, `eigen_seq` et `eigen_gpu`.

## Résultats finaux

> La méthode finale comprends les BLAS de la bibliothèque CUDA et une taille de bloc de 128.

| Methode        | Taille du vecteur | Mflop/s  | Temps (s) |
|:---------------|:-----------------:|:--------:|:---------:|
| CPU Séquentiel |       2048        |   1750   |    0.5    |
| GPU            |       2048        |  86000   |   <0.1    |
| CPU Séquentiel |       32768       |   1750   |   192.3   |
| GPU            |       32768       | 162240.2 |    1.8    |

On a un temps total diminué d'environ 106% en utilisant une Nvidia P100.

### Optimisations effectuées

> Taille des blocs à 256

| Optimisation                                                                          | Mflop/s | Temps (s) | Iterations |
|---------------------------------------------------------------------------------------|:-------:|:---------:|:----------:|
| Accumulation dans une variable temporaire plutot que des accès mémoire du type `Y[i]` | 33 000  |    10     |   ~ 160    |
| Utilisation de la réduction pour `norm`                                               |    /    |     /     |     /      |
| -> avec `atomicAdd` dans `norm`                                                       | 33 000  |    13s    |   ~ 200    |
| -> avec tableau et réduction sur le CPU                                               | 33 000  |    13s    |   ~ 200    |
| Réduction corrigée sur GPU (pour `norm` et `error`)                                   | 33 100  |    10s    |   ~ 160    |
| CuBLAS                                                                                | 160 000 |   1.8s    |   ~ 160    |

On ne remarque pas d'amélioration notable en utilisant les BLAS pour les autres kernels.

### Optimisations possibles non explorées

- Parallelisation du Kernel `vecMatMultKernel` dans chaque ligne plutot que par ligne

### Auto tuning (résultats)

Pour `N = 32768`.

> On se basera sur la version finale optimisée

| Taille des blocs | Mflops/s | Temps (s) |
|:----------------:|:--------:|:---------:|
|        32        | 162091.6 |    1.8    |
|        64        | 161952.2 |    1.8    |
|       128        | 162240.2 |    1.8    |
|       256        | 162111.2 |    1.8    |
|       512        | 162275.7 |    1.8    |
|       1024       | 162128.9 |    1.8    |

On peut observer une augmentation des performances lorsqu'on
augmente la taille des blocs (le plus gros calcul étant fait par CuBLAS), on peut
observer un pic local avec des blocs de 128 (sur les flop/s mais pas sur le temps en secondes qui est trop peu précis).
Pour mieux voir l'impact de la taille des blocs on passera par la version sans les BLAS.

> On se base sur la dernière version sans les blas

| Taille des blocs | Mflops/s | Temps (s) |
|:----------------:|:--------:|:---------:|
|        32        | 24005.6  |   13.7    |
|        64        | 28151.3  |   11.7    |
|       128        | 39324.5  |    8.4    |
|       256        | 33129.2  |   10.0    |
|       512        | 29361.8  |   11.2    |
|       1024       | 33018.1  |   10.0    |

On peut voir une forte augmentation des performances pour une taille de bloc à 128,
qui est donc notre taille optimale de bloc.